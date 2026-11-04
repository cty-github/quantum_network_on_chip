#!/usr/bin/env python3
"""Run the routing-method ablation study and persist reproducible results.

The runner evaluates two ablated strategies on the same benchmark instances:

* ``mech_eqst``: MECH with on-die EQST segments;
* ``qunoc_wo_opt``: QuNoC without coordinated compilation;

The original MECH and complete QuNoC results are supplied by ``main_exp.csv``
when the figures are generated, so this runner does not repeat those cases.

QuNoC strategies reuse package files produced by ``test_layout_scalability.py``.
Results are cached per case under ``evaluate/result/ablation/package_size``
and merged into ``evaluate/result/ablation/package_size/output.csv`` after every completed case.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import pickle
import random
import sys
import traceback
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any, Iterable, Optional


EVALUATE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = EVALUATE_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import numpy as np

    from baseline.MECH.MECHBenchmarks import (
        MECH_EQST_experiments,
        get_benchmark_program,
    )
    from baseline.mech_utils import (
        CHIPLET_LENGTH,
        DEFAULT_INTER_FIDE,
        DEFAULT_INTER_SPARSITY,
        SENSATE_MAX_CAPACITY,
        SENSATE_MAX_LENGTH,
        aligned_logic_gate_fidelity,
        comparison_fab_model,
        data_qubit_num,
        exact_mech_shape,
        get_mech_eqst_circuit_duration,
        sensate_spacing_case_fields,
    )
    from qunoc.element.benchmark import Control, get_program
    from qunoc.element.chiplet import EdgeType
    from qunoc.element.package import Package
    from qunoc.routing.circuit import (
        Alight,
        Board,
        DirectGate,
        Enter,
        Exit,
        GhzPreparation,
        Interchange,
        SwapGate,
        Teleport,
        Transport,
    )
    from qunoc.routing.router import Router
    from qunoc.routing.fidelity import normalized_gate_fidelity, probability_cost
    from evaluate.test_routing_scalability import (
        EQST_CSV_FIELDS,
        _copy_package_eqst_config,
        _package_evaluation_signature,
    )
except ModuleNotFoundError as error:
    raise SystemExit(
        f"Missing Python dependency {error.name!r}. Activate the project's Python "
        "3.10 environment and install requirements.txt plus pyyaml."
    ) from error


# Invalidate statistics only; these unstaged baselines remain explicitly legacy.
# 仅更新统计缓存版本；无阶段边界的基线仍明确采用 legacy，不伪装成 Lindblad 批次。
SCHEMA_VERSION = 3
DEFAULT_SIZES = [40, 60, 80, 100, 120]
QUANTUM_PROGRAMS = ["vqe", "qaoa", "bv", "s_ghz"]
STRATEGIES = ["mech_eqst", "qunoc_wo_opt"]
QUNOC_STRATEGIES = {"qunoc_wo_opt"}
MECH_EQST_GENERATORS = {
    "vqe": "VQE",
    "qft": "QFT",
    "qaoa": "QAOA",
    "bv": "BV",
    "s_ghz": "S_GHZ",
    "s_hamiltonian_simulation": "S_HAMILTONIAN_SIMULATION",
    "s_mermin_bell": "S_MERMIN_BELL",
    "fully_connected": "FULLY_CONNECTED",
    "circular": "CIRCULAR",
    "bv_hard": "BV_HARD",
}

DEFAULT_LAYOUT_ROOT = EVALUATE_ROOT / "result" / "layout_scalability" / "package_size"
DEFAULT_RESULT_ROOT = EVALUATE_ROOT / "result" / "ablation" / "package_size"
DEFAULT_OUTPUT = DEFAULT_RESULT_ROOT / "output.csv"

ROW_FIELDS = [
    "Baseline",
    "PackageSize",
    "QuantumProgram",
    "#DataQubit",
    "#LogicGate",
    "CircuitDuration",
    "GateFidelity",
    "IntraGate",
    "InterGate",
    "QSTDist",
    "MeasNum",
    "TotalFide",
]
PRE_EQST_ROW_FIELDS = ROW_FIELDS.copy()
LEGACY_ROW_FIELDS = [
    "Algorithm" if field == "QuantumProgram" else field for field in PRE_EQST_ROW_FIELDS
]
PRE_LINDBLAD_ROW_FIELDS = ROW_FIELDS + [
    "QSTEffectiveDist", "QSTNegLogFidelity", "TotalNegLogFidelity", "EQSTModel",
]
ROW_FIELDS += EQST_CSV_FIELDS


def _unique(values: Iterable[Any]) -> list[Any]:
    return list(dict.fromkeys(values))


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _sha256_json(value: Any) -> str:
    payload = json.dumps(value, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _qunoc_gate_sequence(program: Any) -> list[tuple[int, int]]:
    return [
        (int(gate.control_qubit), int(gate.target_qubit))
        for gate in program.logic_gates
        if isinstance(gate, Control)
    ]


def _mech_gate_sequence(program: Any) -> list[tuple[int, int]]:
    gates: list[tuple[int, int]] = []
    for block in program:
        for target, count in block.target_counter.items():
            gates.extend([(int(block.control), int(target))] * int(count))
    return gates


def _benchmark_options(args: argparse.Namespace, quantum_program: str) -> dict[str, Any]:
    options = args.benchmark_options.get(quantum_program, {})
    if not isinstance(options, dict):
        raise ValueError(f"benchmark option for {quantum_program} must be a mapping.")
    return options


def _benchmark_metadata(
    quantum_program: str, data_qubits: int, seed: int, args: argparse.Namespace
) -> dict[str, Any]:
    options = _benchmark_options(args, quantum_program)
    _seed_everything(seed)
    qunoc_program = get_program(quantum_program, data_qubits, **options)
    qunoc_gates = _qunoc_gate_sequence(qunoc_program)

    _seed_everything(seed)
    mech_program = get_benchmark_program(quantum_program, data_qubits, **options)
    mech_gates = _mech_gate_sequence(mech_program)

    if qunoc_gates != mech_gates:
        mismatch = next(
            (
                index
                for index, (qunoc_gate, mech_gate) in enumerate(
                    zip(qunoc_gates, mech_gates)
                )
                if qunoc_gate != mech_gate
            ),
            min(len(qunoc_gates), len(mech_gates)),
        )
        raise RuntimeError(
            "MECH and QuNoC generated different benchmark circuits for "
            f"{quantum_program}/{data_qubits} qubits at gate {mismatch}: "
            f"QuNoC has {len(qunoc_gates)} gates and MECH has "
            f"{len(mech_gates)} gates."
        )

    return {
        "logic_gate_num": len(qunoc_gates),
        "benchmark_hash": _sha256_json(qunoc_gates),
    }


def _mapping_entries(mapping: dict[int, tuple[int, int]]) -> list[list[int]]:
    return [
        [int(logic_qubit), int(position[0]), int(position[1])]
        for logic_qubit, position in sorted(mapping.items())
    ]


def _logic_gate_fidelity(
    package: Package,
    logic_gate_num: int,
    intra_gate_num: int,
    inter_gate_num: int,
    qst_distance: int,
) -> float:
    """Return normalized legacy dummy fidelity from physical counts and raw distance.

    输入 package、逻辑门数及物理计数；输出稳定的 legacy 平均门保真度，不使用 Hamiltonian 成本。
    """
    return normalized_gate_fidelity(
        intra_gate_num,
        inter_gate_num,
        probability_cost(package.fab_model.enhanced_fide, qst_distance),
        logic_gate_num,
        package.fab_model.get_intra_fide(package.fab_model.max_length),
        package.fab_model.inter_fide,
    )


def _statistic_dummy_route_details(router: Router) -> dict[str, Any]:
    """Return dummy legacy metrics with three gates per SWAP and additive errors.

    输入 dummy router；输出 raw-distance legacy 统计，保留每次 SWAP 三门计数和稳定负对数误差。
    """
    intra_gate_num = 0
    inter_gate_num = 0
    qst_distance = 0
    meas_num = 0
    total_neglog_fidelity = 0.0

    for action_id in router.dummy_circuit.action_depend_graph:
        if action_id == -1:
            continue
        action = router.dummy_circuit.action_depend_graph.nodes[action_id]["action"]
        if isinstance(action, GhzPreparation):
            for balise, parent in action.ghz_balise_graph:
                if parent is None:
                    continue
                if router.package_graph.edges[balise, parent]["type"] == EdgeType.Intra:
                    intra_gate_num += 1
                else:
                    inter_gate_num += 1
                total_neglog_fidelity += probability_cost(
                    router.package.get_coupler_fide(balise, parent)
                )
        elif isinstance(action, (Board, Alight, Interchange)):
            inter_gate_num += 3
            total_neglog_fidelity += probability_cost(router.package.fab_model.inter_fide, 3)
        elif isinstance(action, Transport):
            qst_distance += action.get_distance()
            total_neglog_fidelity += probability_cost(
                router.package.fab_model.enhanced_fide, action.get_distance()
            )
        elif isinstance(action, Teleport):
            if action.inter_flag:
                inter_gate_num += 1
            else:
                intra_gate_num += 1
            meas_num += 1
            total_neglog_fidelity += probability_cost(router.package.get_coupler_fide(
                action.ancilla_node, action.data_node
            ))
        elif isinstance(action, (Enter, Exit, SwapGate)):
            if action.inter_flag:
                inter_gate_num += 3
            else:
                intra_gate_num += 3
            if isinstance(action, (Enter, Exit)):
                qubit_u, qubit_v = action.platform, action.register
            else:
                qubit_u, qubit_v = action.qubit_u, action.qubit_v
            total_neglog_fidelity += probability_cost(
                router.package.get_coupler_fide(qubit_u, qubit_v), 3
            )
        elif isinstance(action, DirectGate):
            if action.inter_flag:
                inter_gate_num += 1
            else:
                intra_gate_num += 1
            total_neglog_fidelity += probability_cost(router.package.get_coupler_fide(
                action.phy_ctrl_qubit, action.phy_targ_qubit
            ))

    return {
        "intra_gate": intra_gate_num,
        "inter_gate": inter_gate_num,
        "qst_dist": qst_distance,
        "meas_num": meas_num,
        "total_fide": math.exp(-total_neglog_fidelity),
        "qst_effective_dist": float(qst_distance),
        "qst_neglog_fidelity": probability_cost(router.package.fab_model.enhanced_fide, qst_distance),
        "total_neglog_fidelity": total_neglog_fidelity,
        "eqst_model": "legacy",
        "eqst_accounting_policy": "dummy_raw_distance_three_gate_swap_v1",
        "eqst_config": router.package.op_config.get_eqst_config(),
        "eqst_batch_count": None,
    }


def _statistic_dummy_route_results(router: Router) -> tuple[int, int, int, int, float]:
    """Return the compatibility five-tuple from a dummy router's legacy details.

    输入 dummy router；输出历史五元组，保留每次 SWAP 三门和原始距离语义。
    """
    details = _statistic_dummy_route_details(router)
    return tuple(details[name] for name in (
        "intra_gate", "inter_gate", "qst_dist", "meas_num", "total_fide"
    ))


def _layout_path(args: argparse.Namespace, size: int) -> Path:
    return args.layout_root / f"{size}_{args.layout_strategy}" / "package.pkl"


def _validate_package(package: Package, path: Path, size: int, args: argparse.Namespace) -> None:
    mismatches: list[str] = []
    if tuple(package.package_size) != (size, size):
        mismatches.append(f"package_size={tuple(package.package_size)!r}")

    model = package.fab_model
    expected = {
        "compute_geometry": args.geometry,
        "inter_sparsity": args.inter_sparsity,
        "inter_fide": args.inter_fidelity,
        "max_capacity": SENSATE_MAX_CAPACITY,
        "max_length": SENSATE_MAX_LENGTH,
    }
    actual = {name: getattr(model, name) for name in expected}
    for name, expected_value in expected.items():
        actual_value = actual[name]
        if isinstance(expected_value, float):
            matches = math.isclose(
                float(actual_value), expected_value, rel_tol=0.0, abs_tol=1e-12
            )
        else:
            matches = actual_value == expected_value
        if not matches:
            mismatches.append(f"{name}={actual_value!r} (expected {expected_value!r})")

    if mismatches:
        message = f"Package {path} has incompatible parameters: " + "; ".join(mismatches)
        if args.allow_package_parameter_mismatch:
            print(f"WARNING: {message}", file=sys.stderr, flush=True)
        else:
            raise ValueError(
                message
                + ". Regenerate the package or pass "
                "--allow-package-parameter-mismatch."
            )


def _run_qunoc(
    size: int,
    quantum_program: str,
    data_qubits: int,
    benchmark: dict[str, Any],
    case_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    package_path = _layout_path(args, size).resolve()
    package = Package.from_pickle(filepath=str(package_path))
    _validate_package(package, package_path, size, args)
    _copy_package_eqst_config(package, force_legacy=True)

    _seed_everything(args.seed)
    program = get_program(
        quantum_program, data_qubits, **_benchmark_options(args, quantum_program)
    )
    gate_sequence = _qunoc_gate_sequence(program)
    if len(gate_sequence) != benchmark["logic_gate_num"]:
        raise RuntimeError("QuNoC benchmark changed after the equivalence check.")
    if _sha256_json(gate_sequence) != benchmark["benchmark_hash"]:
        raise RuntimeError("QuNoC benchmark hash changed after the equivalence check.")

    router = Router(
        package,
        program,
        route_strategy="immediate",
        result_dir=str(case_dir),
        block_strategy="legacy",
        ghz_source_strategy="control",
        ghz_pairing_strategy="ordered",
        tele_platform_strategy="nearest",
        gate_select_strategy="first",
        gate_order_strategy="as_is",
    )
    circuit = router.dummy_route_program_on_package(args.mapping_strategy)
    details = _statistic_dummy_route_details(router)
    duration_ns = router.estimate_dummy_circuit_duration()

    if args.save_circuits:
        circuit.to_pickle(str(case_dir / "qunoc_wo_opt_circuit.pkl"))

    mapping = _mapping_entries(router.init_l2p_map)
    return {
        "data_qubit_num": data_qubits,
        "logic_gate_num": benchmark["logic_gate_num"],
        "circuit_duration_ns": float(duration_ns),
        "gate_fidelity": _logic_gate_fidelity(
            package,
            benchmark["logic_gate_num"],
            details["intra_gate"],
            details["inter_gate"],
            details["qst_dist"],
        ),
        **details,
        "initial_mapping": mapping,
        "initial_mapping_hash": _sha256_json(mapping),
        "layout_package_path": str(package_path),
    }


def _run_mech(
    size: int,
    quantum_program: str,
    data_qubits: int,
    benchmark: dict[str, Any],
    case_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    package_size = (size, size)
    chiplet_array_dim, chiplet_size = exact_mech_shape(
        package_size, args.chiplet_length
    )
    case_fields = sensate_spacing_case_fields(
        chiplet_size,
        sensate_spacing=args.inter_sparsity,
        geometry=args.geometry,
    )
    kwargs = {
        "structure": args.geometry,
        "chiplet_array_dim": chiplet_array_dim,
        "chiplet_size": chiplet_size,
        "benchmarks": [quantum_program],
        "data_qubit_num": data_qubits,
        "cross_link_sparsity": case_fields["cross_link_sparsity"],
        "cross_link_row_spacing": case_fields["cross_link_row_spacing"],
        "cross_link_col_spacing": case_fields["cross_link_col_spacing"],
        "benchmark_options": {
            quantum_program: _benchmark_options(args, quantum_program)
        },
    }

    _seed_everything(args.seed)
    results, circuits = _run_single_mech_eqst_benchmark(
        quantum_program,
        kwargs,
        args.mech_eqst_basic_time_ns,
    )

    result = results[quantum_program]
    logic_gate_num = int(result["logic_gate_num"])
    if logic_gate_num != benchmark["logic_gate_num"]:
        raise RuntimeError(
            f"mech_eqst compiled {logic_gate_num} logical gates, expected "
            f"{benchmark['logic_gate_num']}."
        )

    duration_ns = get_mech_eqst_circuit_duration(result)

    if args.save_circuits:
        with (case_dir / "mech_eqst_circuit.pkl").open("wb") as file:
            pickle.dump(circuits[quantum_program], file)

    model = comparison_fab_model(
        geometry=args.geometry,
        inter_sparsity=args.inter_sparsity,
        inter_fide=args.inter_fidelity,
    )
    return {
        "data_qubit_num": data_qubits,
        "logic_gate_num": logic_gate_num,
        "circuit_duration_ns": float(duration_ns),
        "gate_fidelity": float(aligned_logic_gate_fidelity(result, model)),
        "intra_gate": int(result.get("on_chip", result.get("on-chip", 0))),
        "inter_gate": int(result.get("cross_chip", result.get("cross-chip", 0))),
        "qst_dist": int(result.get("eqst_dist", 0)),
        "meas_num": int(result.get("meas_num", 0)),
        "total_fide": None,
        "qst_effective_dist": float(result.get("eqst_dist", 0)),
        "qst_neglog_fidelity": probability_cost(model.enhanced_fide, result.get("eqst_dist", 0)),
        # MECH exposes normalized counts, not an exact staged circuit fidelity.
        "total_neglog_fidelity": None,
        "eqst_model": "legacy",
        "eqst_accounting_policy": "mech_eqst_raw_distance_v1",
        "eqst_config": None,
        "eqst_batch_count": None,
        "initial_mapping": None,
        "initial_mapping_hash": None,
        "layout_package_path": None,
        "mech_cross_link_count": int(case_fields["cross_link_sparsity"]),
        "mech_eqst_basic_time_ns": float(args.mech_eqst_basic_time_ns),
    }


def _run_single_mech_eqst_benchmark(
    quantum_program: str,
    kwargs: dict[str, Any],
    basic_time_ns: float,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Avoid eager construction of every unused MECH EQST benchmark."""
    target_generator = MECH_EQST_GENERATORS[quantum_program]
    function_globals = MECH_EQST_experiments.__globals__
    originals: dict[str, Any] = {}

    def empty_benchmark(*_args: Any, **_kwargs: Any) -> list[Any]:
        return []

    try:
        for generator in MECH_EQST_GENERATORS.values():
            if generator == target_generator or generator not in function_globals:
                continue
            originals[generator] = function_globals[generator]
            function_globals[generator] = empty_benchmark
        return MECH_EQST_experiments(
            **kwargs,
            basic_time=basic_time_ns,
        )
    finally:
        function_globals.update(originals)


def _case_directory(
    result_root: Path, size: int, quantum_program: str, strategy: str
) -> Path:
    return result_root / f"{size}_{quantum_program}_{strategy}"


def _signature(
    strategy: str,
    size: int,
    quantum_program: str,
    benchmark: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    signature: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "strategy": strategy,
        "package_size": size,
        "quantum_program": quantum_program,
        "benchmark_options": _benchmark_options(args, quantum_program),
        "seed": args.seed,
        "data_qubit_num": data_qubit_num((size, size)),
        "logic_gate_num": benchmark["logic_gate_num"],
        "benchmark_hash": benchmark["benchmark_hash"],
        "geometry": args.geometry,
        "inter_sparsity": args.inter_sparsity,
        "inter_fidelity": args.inter_fidelity,
        "chiplet_length": args.chiplet_length,
        "eqst_model": "legacy",
    }
    if strategy in QUNOC_STRATEGIES:
        path = _layout_path(args, size).resolve()
        package = Package.from_pickle(filepath=str(path))
        _copy_package_eqst_config(package, force_legacy=True)
        signature.update(
            {
                **_package_evaluation_signature(package, path),
                "layout_strategy": args.layout_strategy,
                "mapping_strategy": args.mapping_strategy,
                "dummy_statistics": "aligned_physical_gate_counts_v1",
                "eqst_accounting_policy": "dummy_raw_distance_three_gate_swap_v1",
            }
        )
    if strategy == "mech_eqst":
        model = comparison_fab_model(
            geometry=args.geometry, inter_sparsity=args.inter_sparsity,
            inter_fide=args.inter_fidelity,
        )
        signature.update({
            "mech_eqst_basic_time_ns": args.mech_eqst_basic_time_ns,
            "eqst_accounting_policy": "mech_eqst_raw_distance_v1",
            "enhanced_fide": float(model.enhanced_fide),
            "normalized_intra_fide": float(model.get_intra_fide(model.max_length)),
        })
    return signature


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2, allow_nan=False)
    temporary.replace(path)


def _normalize_result_schema(result: dict[str, Any]) -> dict[str, Any]:
    if "quantum_program" not in result and "algorithm" in result:
        result["quantum_program"] = result["algorithm"]
    result.pop("algorithm", None)
    signature = result.get("signature")
    if isinstance(signature, dict):
        if "quantum_program" not in signature and "algorithm" in signature:
            signature["quantum_program"] = signature["algorithm"]
        signature.pop("algorithm", None)
    return result


def _load_cached_result(
    path: Path, signature: dict[str, Any]
) -> Optional[dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as file:
            cached = _normalize_result_schema(json.load(file))
    except (OSError, json.JSONDecodeError):
        return None
    if cached.get("status") != "complete" or cached.get("signature") != signature:
        return None
    return cached


def _result_row(result: dict[str, Any]) -> dict[str, Any]:
    """功能 / Purpose: Export aligned baseline metrics with explicit model version / 导出对齐基线指标及明确版本。

    Inputs / 输入: Cached or freshly computed result / 缓存或新计算的结果。
    Outputs / 输出: CSV row; no Monte Carlo claim for legacy baselines / CSV 行，legacy 不声称蒙特卡洛精度。
    """
    result = _normalize_result_schema(result)
    return {
        "Baseline": result["strategy"],
        "PackageSize": int(result["package_size"]),
        "QuantumProgram": result["quantum_program"],
        "#DataQubit": int(result["data_qubit_num"]),
        "#LogicGate": int(result["logic_gate_num"]),
        "CircuitDuration": float(result["circuit_duration_ns"]),
        "GateFidelity": float(result["gate_fidelity"]),
        "IntraGate": int(result["intra_gate"]),
        "InterGate": int(result["inter_gate"]),
        "QSTDist": int(result["qst_dist"]),
        "MeasNum": int(result["meas_num"]),
        "TotalFide": result.get("total_fide"),
        "QSTEffectiveDist": result.get("qst_effective_dist"),
        "QSTNegLogFidelity": result.get("qst_neglog_fidelity"),
        "TotalNegLogFidelity": result.get("total_neglog_fidelity"),
        "EQSTModel": result.get("eqst_model", "legacy"),
        "EQSTVersion": (result.get("eqst_config") or {}).get("version", "legacy-v1"),
        "EQSTMCBatchCount": None,
        "EQSTUnconvergedBatchCount": None,
        "EQSTConfidenceLowerBound": None,
    }


def _row_key(row: dict[str, Any]) -> tuple[int, str, str]:
    return (
        int(row["PackageSize"]),
        str(row["QuantumProgram"]).lower(),
        str(row["Baseline"]).lower(),
    )


def _load_existing_rows(
    path: Path,
) -> tuple[dict[tuple[int, str, str], dict[str, Any]], bool]:
    """功能 / Purpose: Load baseline rows and recognize historical CSV headers / 读取基线行并识别历史表头。

    Inputs / 输入: Existing CSV path / 已有 CSV 路径。
    Outputs / 输出: Indexed rows and whether their header needs migration / 索引行及是否需要迁移表头。
    """
    if not path.is_file():
        return {}, False
    with path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames == ROW_FIELDS:
            migrated = False
        elif reader.fieldnames in (PRE_EQST_ROW_FIELDS, LEGACY_ROW_FIELDS, PRE_LINDBLAD_ROW_FIELDS):
            migrated = True
        elif reader.fieldnames is not None and {
            "PackageSize",
            "Algorithm",
            "Strategy",
            "DataQubit",
            "LogicGate",
        }.issubset(reader.fieldnames):
            migrated = True
        else:
            raise ValueError(
                f"Existing output {path} has incompatible columns. Expected "
                f"{ROW_FIELDS} or the previous ablation schema."
            )

        rows = {}
        for source_row in reader:
            if "Strategy" in source_row:
                row = {
                    "Baseline": source_row["Strategy"],
                    "PackageSize": source_row["PackageSize"],
                    "QuantumProgram": source_row.get(
                        "QuantumProgram", source_row.get("Algorithm", "")
                    ),
                    "#DataQubit": source_row["DataQubit"],
                    "#LogicGate": source_row["LogicGate"],
                    "CircuitDuration": source_row.get("CircuitDuration", ""),
                    "GateFidelity": source_row.get("GateFidelity", ""),
                    "IntraGate": source_row.get("IntraGate", ""),
                    "InterGate": source_row.get("InterGate", ""),
                    "QSTDist": source_row.get("QSTDist", ""),
                    "MeasNum": source_row.get("MeasNum", ""),
                    "TotalFide": source_row.get("TotalFide", ""),
                }
            else:
                row = {field: source_row.get(field, "") for field in ROW_FIELDS}
                if not row.get("QuantumProgram"):
                    row["QuantumProgram"] = source_row.get("Algorithm", "")
            if migrated:
                # Historical ablation rows used raw-distance legacy accounting.
                # Do not manufacture negative-log metrics from underflowed totals.
                row["EQSTModel"] = "legacy"
                row["QSTEffectiveDist"] = row.get("QSTDist", "")
            rows[_row_key(row)] = row
        return rows, migrated


def _write_rows(
    path: Path,
    rows: dict[tuple[int, str, str], dict[str, Any]],
) -> None:
    quantum_program_order = {name: index for index, name in enumerate(QUANTUM_PROGRAMS)}
    strategy_order = {name: index for index, name in enumerate(STRATEGIES)}

    def sort_key(row: dict[str, Any]) -> tuple[int, int, int]:
        return (
            int(row["PackageSize"]),
            quantum_program_order.get(str(row["QuantumProgram"]).lower(), len(QUANTUM_PROGRAMS)),
            strategy_order.get(str(row["Baseline"]).lower(), len(STRATEGIES)),
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=ROW_FIELDS)
        writer.writeheader()
        for row in sorted(rows.values(), key=sort_key):
            writer.writerow({field: row.get(field, "") for field in ROW_FIELDS})
    temporary.replace(path)


def _validate_cross_strategy_results(
    results: dict[tuple[int, str, str, int], dict[str, Any]],
    sizes: list[int],
    quantum_programs: list[str],
    seed: int,
) -> None:
    for size, quantum_program in product(sizes, quantum_programs):
        selected = [
            result
            for result in results.values()
            if int(result["package_size"]) == size
            and str(result["quantum_program"]).lower() == quantum_program
            and int(result["seed"]) == seed
        ]
        if len(selected) < 2:
            continue
        gate_counts = {int(result["logic_gate_num"]) for result in selected}
        benchmark_hashes = {str(result["benchmark_hash"]) for result in selected}
        if len(gate_counts) != 1 or len(benchmark_hashes) != 1:
            raise RuntimeError(
                f"Cross-strategy benchmark mismatch for {size}/{quantum_program}: "
                f"gate counts={sorted(gate_counts)}, hashes={sorted(benchmark_hashes)}"
            )

        qunoc_mapping_hashes = {
            str(result["initial_mapping_hash"])
            for result in selected
            if str(result["strategy"]) in QUNOC_STRATEGIES
            and result.get("initial_mapping_hash") not in {None, ""}
        }
        if len(qunoc_mapping_hashes) > 1:
            raise RuntimeError(
                f"QuNoC initial mappings differ for {size}/{quantum_program}: "
                f"{sorted(qunoc_mapping_hashes)}"
            )


def _preflight(args: argparse.Namespace) -> None:
    if getattr(args, "eqst_model", "legacy") != "legacy":
        raise ValueError(
            "Ablation strategies mech_eqst and qunoc_wo_opt use legacy accounting: "
            "their MECH/dummy circuits do not provide staged Hamiltonian batches. "
            "Use evaluate/test_routing_scalability.py --eqst-model hamiltonian "
            "for full QuNoC routing."
        )
    if any(size <= 0 for size in args.sizes):
        raise ValueError("Package sizes must be positive integers.")
    if args.chiplet_length <= 0:
        raise ValueError("--chiplet-length must be positive.")
    if args.inter_sparsity <= 0:
        raise ValueError("--inter-sparsity must be positive.")
    if not 0 < args.inter_fidelity <= 1:
        raise ValueError("--inter-fidelity must be in (0, 1].")
    if not 0 <= args.seed < 2**32:
        raise ValueError("--seed must be in [0, 2**32).")
    if args.mech_eqst_basic_time_ns <= 0:
        raise ValueError("--mech-eqst-basic-time-ns must be positive.")

    if QUNOC_STRATEGIES.intersection(args.strategies):
        missing = [
            str(_layout_path(args, size))
            for size in args.sizes
            if not _layout_path(args, size).is_file()
        ]
        if missing:
            formatted = "\n  ".join(missing)
            raise FileNotFoundError(
                "Missing QuNoC layout packages:\n  "
                + formatted
                + "\nGenerate the selected package-size layouts with "
                "evaluate/test_layout_scalability.py before running this experiment."
            )
        for size in args.sizes:
            path = _layout_path(args, size).resolve()
            package = Package.from_pickle(filepath=str(path))
            _validate_package(package, path, size, args)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MECH/QuNoC routing-method ablation experiments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""examples:
  python evaluate/run_ablation.py --sizes 40 --quantum-programs bv
  python evaluate/run_ablation.py --sizes 40 60 80 100 120 \\
      --quantum-programs vqe qft qaoa bv --keep-going
  python evaluate/run_ablation.py --dry-run
""",
    )
    parser.add_argument("--sizes", nargs="+", type=int, default=DEFAULT_SIZES)
    parser.add_argument(
        "--quantum-programs", nargs="+", choices=QUANTUM_PROGRAMS, default=QUANTUM_PROGRAMS
    )
    parser.add_argument(
        "--strategies", nargs="+", choices=STRATEGIES, default=STRATEGIES
    )
    parser.add_argument(
        "--eqst-model", choices=["legacy", "hamiltonian"], default="legacy",
        help="Ablation uses legacy accounting; Hamiltonian is rejected (use full routing scalability).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--layout-root",
        type=Path,
        default=DEFAULT_LAYOUT_ROOT,
        help="Directory containing <size>_<strategy>/package.pkl layouts.",
    )
    parser.add_argument("--layout-strategy", default="heuristic")
    parser.add_argument(
        "--mapping-strategy", choices=["random", "sabre"], default="random"
    )
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--chiplet-length", type=int, default=CHIPLET_LENGTH)
    parser.add_argument(
        "--geometry",
        choices=["square", "hexagon", "heavy_square", "heavy_hexagon"],
        default="square",
    )
    parser.add_argument(
        "--inter-sparsity", type=int, default=DEFAULT_INTER_SPARSITY
    )
    parser.add_argument(
        "--inter-fidelity", type=float, default=DEFAULT_INTER_FIDE
    )
    parser.add_argument(
        "--benchmark-options",
        type=str,
        default='{"qaoa":{"problem_graph":"QAOAcycle","order_strategy":"control_grouped","seed":52}}',
        help="JSON mapping of per-benchmark options, e.g. '{\"qaoa\": {\"problem_graph\": \"cycle\"}}'.",
    )
    parser.add_argument(
        "--mech-eqst-basic-time-ns",
        type=float,
        default=1000.0,
        help="Basic non-EQST stage duration used by MECH_EQST_experiments.",
    )
    parser.add_argument("--save-circuits", action="store_true")
    parser.add_argument(
        "--allow-package-parameter-mismatch", action="store_true"
    )
    parser.add_argument(
        "--force", action="store_true", help="Ignore matching per-case caches."
    )
    parser.add_argument(
        "--keep-going", action="store_true", help="Continue after a failed case."
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Validate inputs and list cases only."
    )
    args = parser.parse_args()
    args.benchmark_options = json.loads(args.benchmark_options)
    if not isinstance(args.benchmark_options, dict):
        raise ValueError("--benchmark-options must be a JSON mapping.")
    args.sizes = _unique(args.sizes)
    args.quantum_programs = _unique(args.quantum_programs)
    args.strategies = _unique(args.strategies)
    args.layout_root = args.layout_root.expanduser().resolve()
    args.result_root = args.result_root.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    return args


def main() -> int:
    args = _parse_args()
    _preflight(args)

    cases = list(product(args.sizes, args.quantum_programs, args.strategies))
    print(
        f"Ablation matrix: {len(args.sizes)} sizes x {len(args.quantum_programs)} "
        f"quantum programs x {len(args.strategies)} strategies = {len(cases)} cases",
        flush=True,
    )
    print(f"Output CSV: {args.output}", flush=True)
    print(f"Raw results: {args.result_root}", flush=True)
    if args.dry_run:
        for size, quantum_program in product(args.sizes, args.quantum_programs):
            benchmark = _benchmark_metadata(
                quantum_program, data_qubit_num((size, size)), args.seed, args
            )
            print(
                f"  benchmark {size}x{size} {quantum_program}: "
                f"{benchmark['logic_gate_num']} gates, "
                f"sha256={benchmark['benchmark_hash'][:12]}..."
            )
        for size, quantum_program, strategy in cases:
            print(f"  case {size}x{size} {quantum_program} {strategy}")
        return 0

    rows, migrated = _load_existing_rows(args.output)
    if migrated:
        _write_rows(args.output, rows)
        print(f"Migrated output CSV to the MECH baseline schema: {args.output}", flush=True)
    benchmark_cache: dict[tuple[int, str], dict[str, Any]] = {}
    completed_results: dict[tuple[int, str, str, int], dict[str, Any]] = {}
    failures: list[str] = []

    for index, (size, quantum_program, strategy) in enumerate(cases, start=1):
        label = f"{size}x{size}/{quantum_program}/{strategy}"
        case_dir = _case_directory(args.result_root, size, quantum_program, strategy)
        case_dir.mkdir(parents=True, exist_ok=True)
        result_path = case_dir / "exp_result.json"
        failure_path = case_dir / "failure.json"

        try:
            benchmark_key = (size, quantum_program)
            if benchmark_key not in benchmark_cache:
                benchmark_cache[benchmark_key] = _benchmark_metadata(
                    quantum_program, data_qubit_num((size, size)), args.seed, args
                )
            benchmark = benchmark_cache[benchmark_key]
            signature = _signature(strategy, size, quantum_program, benchmark, args)

            cached = None if args.force else _load_cached_result(result_path, signature)
            if cached is not None:
                result = cached
                print(f"[{index}/{len(cases)}] cached {label}", flush=True)
            else:
                print(f"[{index}/{len(cases)}] running {label}", flush=True)
                if strategy in QUNOC_STRATEGIES:
                    metrics = _run_qunoc(
                        size,
                        quantum_program,
                        data_qubit_num((size, size)),
                        benchmark,
                        case_dir,
                        args,
                    )
                else:
                    metrics = _run_mech(
                        size,
                        quantum_program,
                        data_qubit_num((size, size)),
                        benchmark,
                        case_dir,
                        args,
                    )
                result = {
                    "status": "complete",
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                    "signature": signature,
                    "package_size": size,
                    "quantum_program": quantum_program,
                    "strategy": strategy,
                    "seed": args.seed,
                    "benchmark_hash": benchmark["benchmark_hash"],
                    **metrics,
                }
                _write_json(result_path, result)
                print(
                    f"[{index}/{len(cases)}] done {label}: "
                    f"duration={result['circuit_duration_ns'] / 1_000_000:.6g} ms, "
                    f"fidelity={result['gate_fidelity']:.6g}",
                    flush=True,
                )

            result_key = (size, quantum_program, strategy, args.seed)
            completed_results[result_key] = result
            row = _result_row(result)
            rows[_row_key(row)] = row
            _validate_cross_strategy_results(
                completed_results, [size], [quantum_program], args.seed
            )
            _write_rows(args.output, rows)
            if failure_path.exists():
                failure_path.unlink()
        except Exception as error:
            failure = {
                "status": "failed",
                "failed_at": datetime.now(timezone.utc).isoformat(),
                "case": label,
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            }
            _write_json(failure_path, failure)
            failures.append(f"{label}: {type(error).__name__}: {error}")
            print(f"[{index}/{len(cases)}] FAILED {failures[-1]}", file=sys.stderr, flush=True)
            if not args.keep_going:
                raise

    _validate_cross_strategy_results(
        completed_results, args.sizes, args.quantum_programs, args.seed
    )
    _write_rows(args.output, rows)
    if failures:
        print("Completed with failed cases:", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        return 1

    print(f"Completed {len(cases)} ablation cases.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
