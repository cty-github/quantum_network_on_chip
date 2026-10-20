import argparse
import copy
import csv
import hashlib
import json
import logging
import math
import os
import sys
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Any

sys.path.append(str(Path(__file__).resolve().parents[1]))

import yaml

from qunoc.element.benchmark import get_program
from qunoc.element.package import Package
from qunoc.routing.circuit import Circuit
from qunoc.routing.router import Router


@dataclass
class ExperimentSpec:
    name: str
    exp_dir: str
    log_filename: str
    output_csv: str
    defaults: dict[str, Any] = field(default_factory=dict)
    scalability: dict[str, list[Any]] = field(default_factory=dict)


@dataclass
class ExperimentConfig:
    exp_dir: str = "./result/routing_scalability"
    log_filename: str = "log"
    output_csv: str = "output.csv"
    defaults: dict[str, Any] = field(default_factory=dict)
    experiments: dict[str, ExperimentSpec] = field(default_factory=dict)


DEFAULT_CONFIG: dict[str, Any] = {
    "exp_dir": "./result/routing_scalability",
    "log_filename": "log",
    "output_csv": "output.csv",
    "defaults": {
        "layout_root_dir": "./result/layout_scalability",
        "quantum_programs": ["vqe", "qaoa", "bv", "s_ghz"],
        "layout_design_strategy": "greedy",
        "package_size": [100, 100],
        "package_area": 10000,
        "reuse_result": True,
        "eqst": {"model": "legacy"},
    },
    "experiments": {
        "package_size": {
            "exp_dir": "./result/routing_scalability/package_size",
            "output_csv": "output.csv",
            "defaults": {},
            "scalability": {
                "package_size_to_test": [40, 60, 80, 100, 120],
            },
        },
        "package_shape": {
            "exp_dir": "./result/routing_scalability/package_shape",
            "output_csv": "output.csv",
            "defaults": {},
            "scalability": {
                "package_x_length_to_test": [10, 20, 40, 50, 80, 100],
            },
        },
        "geometry": {
            "exp_dir": "./result/routing_scalability/geometry",
            "output_csv": "output.csv",
            "defaults": {},
            "scalability": {
                "geometry_to_test": ["square", "hexagon", "heavy_square", "heavy_hexagon"],
            },
        },
        "inter_fidelity": {
            "exp_dir": "./result/routing_scalability/inter_fidelity",
            "output_csv": "output.csv",
            "defaults": {},
            "scalability": {
                "inter_fidelity_to_test": [0.96, 0.965, 0.97, 0.975, 0.98, 0.985, 0.99],
            },
        },
        "chiplet_sparsity": {
            "exp_dir": "./result/routing_scalability/chiplet_sparsity",
            "output_csv": "output.csv",
            "defaults": {},
            "scalability": {
                "chiplet_sparsity_to_test": [1, 3, 5, 7],
            },
        },
    },
}

AXIS_TARGETS = {
    "layout_design_strategy_to_test": "layout_design_strategy",
    "package_size_to_test": "package_size",
    "package_x_length_to_test": "package_x_size",
    "geometry_to_test": "fab_model.compute_geometry",
    "inter_fidelity_to_test": "fab_model.inter_fide",
    "chiplet_sparsity_to_test": "fab_model.inter_sparsity",
}

AXIS_LABELS = {
    "layout_design_strategy_to_test": "LayoutDesignStrategy",
    "package_size_to_test": "PackageSize",
    "package_x_length_to_test": "PackageXLength",
    "geometry_to_test": "PackageGeometry",
    "inter_fidelity_to_test": "InterFidelity",
    "chiplet_sparsity_to_test": "ChipletSparsity",
}

# Bump when evaluation/cache semantics change; old linear-only results are not reusable.
EVALUATION_SCHEMA_VERSION = 3
COMPILATION_SCHEMA_VERSION = 1
EQST_OPTION_NAMES = (
    "model",
    "reference_time_ns",
    "reference_distance",
    "g_max",
    "max_segment_steps",
    "beam_width",
)
EQST_CSV_FIELDS = [
    "QSTEffectiveDist",
    "QSTNegLogFidelity",
    "TotalNegLogFidelity",
    "EQSTModel",
]

ROUTING_OPTION_DEFAULTS: dict[str, Any] = {
    "route_strategy": "immediate",
    "mapping_strategy": "random",
    "block_strategy": "legacy",
    "move_min_gate_per_migrated_qubit": 3.0,
    "move_cost_ratio": 0.75,
    "move_density_divisor": 2.0,
    "ghz_source_strategy": "control",
    "ghz_pairing_strategy": "ordered",
    "tele_platform_strategy": "nearest",
    "gate_select_strategy": "first",
    "gate_order_strategy": "as_is",
}


def _deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_update(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _copy_package_eqst_config(
    package: Package,
    options: dict[str, Any] | None = None,
    *,
    force_legacy: bool = False,
) -> dict[str, Any]:
    """Apply EQST options to an isolated operation config and return JSON settings.

    输入 package、短键配置和 legacy 强制标志；复制后覆盖，返回可序列化配置，避免共享修改。
    """
    options = {} if options is None else options
    if not isinstance(options, dict):
        raise ValueError("eqst must be a mapping.")
    unknown = set(options) - set(EQST_OPTION_NAMES)
    if unknown:
        raise ValueError(f"Unknown eqst settings: {sorted(unknown)}")
    package.op_config = copy.deepcopy(package.op_config)
    # Never inherit an opt-in model silently from an existing package artifact.
    package.op_config.eqst_model = "legacy"
    for name, value in options.items():
        setattr(package.op_config, f"eqst_{name}", value)
    if force_legacy:
        package.op_config.eqst_model = "legacy"
    return package.op_config.get_eqst_config()


def _add_eqst_arguments(parser: argparse.ArgumentParser) -> None:
    """Add optional model/calibration flags to a parser; return None.

    输入命令行解析器，添加 EQST 模型、标定及规划参数；无返回值。
    """
    parser.add_argument(
        "--eqst-model", choices=["legacy", "hamiltonian"], default=None,
        help="EQST model for full QuNoC routes (default: legacy); baselines stay legacy.",
    )
    for name, value_type, help_text in (
        ("reference_time_ns", float, "Hamiltonian reference transfer time in ns (default: 125)."),
        ("reference_distance", float, "Hamiltonian reference distance (default: 5)."),
        ("g_max", float, "Maximum Hamiltonian coupling in rad/ns (default: 3*pi/250)."),
        ("max_segment_steps", int, "Heuristic planner stage limit (default: 10); fallback may exceed it."),
        ("beam_width", int, "Hamiltonian batch planner beam width (default: 128)."),
    ):
        parser.add_argument(
            "--eqst-" + name.replace("_", "-"), type=value_type, default=None,
            help=help_text,
        )


def _eqst_cli_overrides(args: argparse.Namespace) -> dict[str, Any]:
    """Return only explicitly supplied EQST CLI settings from parsed arguments.

    输入已解析命令行参数，输出显式指定的短键配置，避免覆盖 YAML 中未指定的值。
    """
    return {
        name: getattr(args, f"eqst_{name}")
        for name in EQST_OPTION_NAMES
        if getattr(args, f"eqst_{name}", None) is not None
    }


def _package_evaluation_signature(package: Package, package_path: str | Path) -> dict[str, Any]:
    """Return JSON-ready package identity, effective EQST, timing and fidelity inputs.

    输入 package 及其文件路径；输出缓存签名，防止旧模型、标定或 package 结果误复用。
    """
    path = Path(package_path).resolve()
    stat = path.stat()
    op_config = package.op_config
    model = package.fab_model
    return {
        "evaluation_schema_version": EVALUATION_SCHEMA_VERSION,
        "layout_package_path": str(path),
        "layout_package_size_bytes": stat.st_size,
        "layout_package_mtime_ns": stat.st_mtime_ns,
        "layout_package_ctime_ns": stat.st_ctime_ns,
        "layout_package_device": stat.st_dev,
        "layout_package_inode": stat.st_ino,
        "eqst_config": op_config.get_eqst_config(),
        "operation_timing_ns": {
            name: float(getattr(op_config, name))
            for name in (
                "single_time", "intra_time", "inter_time", "qst_time_constant",
                "measure_time", "reset_time",
            )
        },
        "fabrication": {
            "max_capacity": int(model.max_capacity),
            "max_length": int(model.max_length),
            "compute_geometry": str(model.compute_geometry),
            "inter_sparsity": int(model.inter_sparsity),
            "single_fide": float(model.single_fide),
            "intra_fide_params": [float(value) for value in model.intra_fide_params],
            "normalized_intra_fide": float(model.get_intra_fide(model.max_length)),
            "inter_fide": float(model.inter_fide),
            "enhanced_fide": float(model.enhanced_fide),
        },
    }


def _package_topology_digest(package: Package) -> str:
    """Hash physical placement/connectivity without timing or fidelity calibration.

    输入 package；输出拓扑摘要，区分布局变化但允许只改标定后重新统计已有电路。
    """
    def chiplet_key(chiplet: Any) -> tuple[Any, ...] | None:
        """Return chiplet geometry only / 输入裸片，返回不含保真度的几何标识。"""
        if chiplet is None:
            return None
        return (type(chiplet).__name__, *(
            getattr(chiplet, name, None)
            for name in ("axis", "pos", "start", "end", "left", "top", "right", "bottom", "geometry")
        ))

    graph = package.package_graph
    nodes = []
    for node, attrs in sorted(graph.nodes(data=True)):
        qubit = attrs.get("qubit")
        nodes.append((
            node, str(attrs.get("type")), str(getattr(qubit, "role", None)),
            chiplet_key(getattr(qubit, "chiplet", None)),
        ))
    edges = sorted(
        (min(source, target), max(source, target), str(attrs.get("type")))
        for source, target, attrs in graph.edges(data=True)
    )
    payload = {"shape": list(package.package_size), "nodes": nodes, "edges": edges}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _compiled_circuit_identity(path: str | Path) -> dict[str, Any]:
    """Return identity/stat fields for a compiled artifact without loading it.

    输入已保存电路路径；输出身份和文件状态，防止将另一份电路与旧统计混用。
    """
    path = Path(path).resolve()
    stat = path.stat()
    return {
        "path": str(path), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
        "ctime_ns": stat.st_ctime_ns, "device": stat.st_dev, "inode": stat.st_ino,
    }


def _legacy_compilation_metadata_matches(
    cached: dict[str, Any], compilation_signature: dict[str, Any]
) -> bool:
    """Check available old-cache metadata for explicit statistics-only migration.

    输入旧结果与当前编译签名；输出是否具备一致元数据，不能证明旧 package 拓扑未变。
    """
    program = cached.get("quantum_program", cached.get("algorithm"))
    path = cached.get("layout_package_path")
    return (
        program == compilation_signature["quantum_program"]
        and isinstance(path, str)
        and str(Path(path).resolve()) == compilation_signature["layout_package_path"]
        and cached.get("data_qubit_num") == compilation_signature["data_qubit_num"]
        and cached.get("routing_options") == compilation_signature["routing_options"]
        and cached.get("program_options", {}) == compilation_signature["program_options"]
        and isinstance(cached.get("logic_gate_num"), int)
        and not isinstance(cached["logic_gate_num"], bool)
        and cached["logic_gate_num"] > 0
    )


def _normalize_package_size(value: Any) -> tuple[int, int]:
    if isinstance(value, int):
        return value, value
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return int(value[0]), int(value[1])
    raise ValueError(f"Invalid package size value: {value!r}")


def _format_value(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return "x".join(str(item) for item in value)
    return str(value)


def _resolve_output_path(path_value: str | os.PathLike[str], *, base_dir: Path) -> str:
    path = Path(path_value)
    if not path.is_absolute():
        path = base_dir / path
    return str(path.resolve())



def _set_nested(config: dict[str, Any], key_path: str, value: Any) -> None:
    parts = key_path.split(".")
    current = config
    for part in parts[:-1]:
        current = current.setdefault(part, {})
    current[parts[-1]] = value


def load_experiment_config(path: str | Path | None) -> ExperimentConfig:
    default_raw = copy.deepcopy(DEFAULT_CONFIG)
    raw = default_raw
    config_dir = Path(__file__).resolve().parent
    if path is not None:
        config_path = Path(path).resolve()
        config_dir = config_path.parent
        with config_path.open("r", encoding="utf-8") as file:
            if config_path.suffix.lower() in {".yaml", ".yml"}:
                user_raw = yaml.safe_load(file) or {}
            else:
                user_raw = json.load(file) or {}
        if not isinstance(user_raw, dict):
            raise ValueError("Config file must contain a mapping at the top level.")
        # If user provided an explicit `experiments` section, replace defaults' experiments
        user_exps = copy.deepcopy(user_raw.get("experiments")) if isinstance(user_raw, dict) else None
        # Merge other keys but avoid merging default experiments when user specified experiments
        user_no_exps = {k: v for k, v in user_raw.items() if k != "experiments"}
        raw = _deep_update(default_raw, user_no_exps)
        if user_exps is not None:
            raw["experiments"] = copy.deepcopy(user_exps)

    root_exp_dir = _resolve_output_path(
        raw.get("exp_dir", DEFAULT_CONFIG["exp_dir"]),
        base_dir=config_dir,
    )
    defaults = copy.deepcopy(raw.get("defaults", {}))
    if "layout_root_dir" in defaults:
        defaults["layout_root_dir"] = _resolve_output_path(
            defaults["layout_root_dir"],
            base_dir=config_dir,
        )

    experiments: dict[str, ExperimentSpec] = {}
    default_experiments = DEFAULT_CONFIG.get("experiments", {})
    for experiment_name, experiment_raw in raw.get("experiments", {}).items():
        if not isinstance(experiment_raw, dict):
            continue
        experiment_override = copy.deepcopy(experiment_raw)
        for optional_section in ("defaults", "scalability"):
            if experiment_override.get(optional_section) is None:
                experiment_override.pop(optional_section, None)
        experiment_config = _deep_update(
            copy.deepcopy(default_experiments.get(experiment_name, {})),
            experiment_override,
        )
        experiment_exp_dir = experiment_config.get(
            "exp_dir",
            os.path.join(root_exp_dir, experiment_name),
        )
        experiments[experiment_name] = ExperimentSpec(
            name=experiment_name,
            exp_dir=_resolve_output_path(experiment_exp_dir, base_dir=config_dir),
            log_filename=str(experiment_config.get("log_filename", raw.get("log_filename", DEFAULT_CONFIG["log_filename"]))),
            output_csv=str(experiment_config.get("output_csv", raw.get("output_csv", DEFAULT_CONFIG["output_csv"]))),
            defaults=copy.deepcopy(experiment_config.get("defaults", {})),
            scalability=copy.deepcopy(experiment_config.get("scalability", {})),
        )

    return ExperimentConfig(
        exp_dir=root_exp_dir,
        log_filename=str(raw.get("log_filename", DEFAULT_CONFIG["log_filename"])),
        output_csv=str(raw.get("output_csv", DEFAULT_CONFIG["output_csv"])),
        defaults=defaults,
        experiments=experiments,
    )


def _build_sweep_axes(scalability: dict[str, list[Any]]) -> tuple[list[str], list[list[tuple[str, Any]]]]:
    axis_names: list[str] = []
    axis_values: list[list[tuple[str, Any]]] = []

    for raw_key, values in scalability.items():
        if not raw_key.endswith("_to_test"):
            continue

        target_key = AXIS_TARGETS.get(raw_key, raw_key[: -len("_to_test")])
        normalized_values: list[Any] = []
        for value in values:
            if target_key == "package_size":
                normalized_values.append(int(value))
            elif target_key == "package_x_size":
                normalized_values.append(int(value))
            elif target_key == "layout_design_strategy":
                normalized_values.append(str(value))
            elif target_key == "fab_model.compute_geometry":
                normalized_values.append(str(value))
            elif target_key == "fab_model.inter_fide":
                normalized_values.append(float(value))
            elif target_key == "fab_model.inter_sparsity":
                normalized_values.append(int(value))
            else:
                normalized_values.append(value)

        axis_names.append(raw_key)
        axis_values.append([(target_key, value) for value in normalized_values])

    return axis_names, axis_values


def _resolve_package_size(run_config: dict[str, Any]) -> tuple[int, int]:
    if "package_x_size" in run_config:
        package_x_size = int(run_config["package_x_size"])
        package_area = int(run_config.get("package_area", 10000))
        if package_x_size <= 0:
            raise ValueError("package_x_size must be positive.")
        if package_area % package_x_size != 0:
            raise ValueError(
                f"package_area {package_area} is not divisible by package_x_size {package_x_size}."
            )
        return package_x_size, package_area // package_x_size
    if "package_size" in run_config:
        return _normalize_package_size(run_config["package_size"])
    raise KeyError("run_config must define package_size or package_x_size.")


def _data_qubit_num(package_size: tuple[int, int]) -> int:
    node_num = math.prod(package_size)
    return int(100 * 2 ** math.log(node_num / 1600, 5))


def _build_layout_package_path(common_defaults: dict[str, Any], spec_name: str, run_config: dict[str, Any]) -> str:
    layout_root_dir = common_defaults.get("layout_root_dir", "./result/layout_scalability")
    layout_dir = os.path.join(layout_root_dir, spec_name)
    layout_strategy = str(run_config.get("layout_design_strategy", common_defaults.get("layout_design_strategy", "greedy")))
    package_size = _resolve_package_size(run_config)
    if spec_name == "package_size":
        return os.path.join(layout_dir, f"{package_size[0]}_{layout_strategy}", "package.pkl")
    if spec_name == "package_shape":
        return os.path.join(layout_dir, f"{package_size[0]}_{package_size[1]}_{layout_strategy}", "package.pkl")
    if spec_name == "geometry":
        geometry = str(run_config.get("fab_model", {}).get("compute_geometry", "square"))
        return os.path.join(layout_dir, f"{geometry}_{layout_strategy}", "package.pkl")
    if spec_name == "inter_fidelity":
        inter_fide = run_config.get("fab_model", {}).get("inter_fide")
        return os.path.join(layout_dir, f"{inter_fide}_{layout_strategy}", "package.pkl")
    if spec_name == "chiplet_sparsity":
        inter_sparsity = run_config.get("fab_model", {}).get("inter_sparsity")
        return os.path.join(layout_dir, f"{inter_sparsity}_{layout_strategy}", "package.pkl")

    raise ValueError(f"Unsupported routing experiment: {spec_name}")


def _resolve_routing_options(
    run_config: dict[str, Any], quantum_program: str
) -> dict[str, Any]:
    routing_options = copy.deepcopy(ROUTING_OPTION_DEFAULTS)
    for key in ROUTING_OPTION_DEFAULTS:
        if key in run_config:
            routing_options[key] = copy.deepcopy(run_config[key])
    routing_options = _deep_update(routing_options, run_config.get("routing", {}))
    routing_options = _deep_update(
        routing_options,
        run_config.get("program_routing", {}).get(quantum_program, {}),
    )
    return routing_options


def _resolve_program_options(
    run_config: dict[str, Any], quantum_program: str
) -> dict[str, Any]:
    program_options = copy.deepcopy(run_config.get("program_options", {}).get(quantum_program, {}))
    if not isinstance(program_options, dict):
        raise ValueError(f"program_options.{quantum_program} must be a mapping.")
    return program_options


def _build_result_dir_name(spec_name: str, run_config: dict[str, Any], quantum_program: str) -> str:
    layout_strategy = str(run_config.get("layout_design_strategy", "greedy"))
    package_size = _resolve_package_size(run_config)
    if spec_name == "package_size":
        return f"{package_size[0]}_{layout_strategy}_{quantum_program}"
    if spec_name == "package_shape":
        return f"{package_size[0]}_{package_size[1]}_{layout_strategy}_{quantum_program}"
    if spec_name == "geometry":
        geometry = str(run_config.get("fab_model", {}).get("compute_geometry", "square"))
        return f"{geometry}_{layout_strategy}_{quantum_program}"
    if spec_name == "inter_fidelity":
        inter_fide = run_config.get("fab_model", {}).get("inter_fide")
        return f"{inter_fide}_{layout_strategy}_{quantum_program}"
    if spec_name == "chiplet_sparsity":
        inter_sparsity = run_config.get("fab_model", {}).get("inter_sparsity")
        return f"{inter_sparsity}_{layout_strategy}_{quantum_program}"
    return f"{layout_strategy}_{quantum_program}"


def _route_once(
    common_defaults: dict[str, Any],
    spec: ExperimentSpec,
    run_config: dict[str, Any],
    quantum_program: str,
) -> dict[str, Any]:
    package_path = _build_layout_package_path(common_defaults, spec.name, run_config)
    if not os.path.exists(package_path):
        raise FileNotFoundError(
            f"Missing exact layout package: {package_path}. "
            "Run evaluate/test_layout_scalability.py with matching layout parameters first."
        )

    package = Package.from_pickle(filepath=package_path)
    _copy_package_eqst_config(package, run_config.get("eqst"))
    package_size = package.package_size
    data_qubit_num = _data_qubit_num(package_size)
    quantum_program = str(quantum_program)
    routing_options = _resolve_routing_options(run_config, quantum_program)
    program_options = _resolve_program_options(run_config, quantum_program)
    signature = {
        **_package_evaluation_signature(package, package_path),
        "quantum_program": quantum_program,
        "data_qubit_num": data_qubit_num,
        "routing_options": routing_options,
        "program_options": program_options,
    }

    compilation_signature = {
        "compilation_schema_version": COMPILATION_SCHEMA_VERSION,
        "layout_package_path": str(Path(package_path).resolve()),
        "package_topology_sha256": _package_topology_digest(package),
        "package_size": list(package_size),
        "quantum_program": quantum_program,
        "data_qubit_num": data_qubit_num,
        "routing_options": routing_options,
        "program_options": program_options,
    }
    statistics_only = bool(run_config.get("statistics_only", False))
    recompile = bool(run_config.get("recompile", False))
    if statistics_only and recompile:
        raise ValueError("statistics_only and recompile cannot both be enabled.")
    result_dir = os.path.join(spec.exp_dir, _build_result_dir_name(spec.name, run_config, quantum_program))
    os.makedirs(result_dir, exist_ok=True)

    cache_path = os.path.join(result_dir, "exp_result.json")
    circuit_path = os.path.join(result_dir, "qunoc_circuit.pkl")
    try:
        with open(cache_path, "r", encoding="utf-8") as file:
            cached = json.load(file)
        if not isinstance(cached, dict):
            cached = {}
    except (OSError, json.JSONDecodeError):
        cached = {}
    circuit_identity = _compiled_circuit_identity(circuit_path) if os.path.isfile(circuit_path) else None
    compilation_matches = cached.get("compilation_signature") == compilation_signature
    artifact_matches = circuit_identity is not None and cached.get("compiled_circuit_identity") == circuit_identity
    legacy_reuse = (
        statistics_only and "compilation_signature" not in cached
        and circuit_identity is not None
        and _legacy_compilation_metadata_matches(cached, compilation_signature)
    )
    reusable_circuit = (compilation_matches and artifact_matches) or legacy_reuse
    if (
        not statistics_only and not recompile
        and bool(run_config.get("reuse_result", True))
        and reusable_circuit and cached.get("signature") == signature
    ):
        return cached
    if statistics_only and not reusable_circuit:
        raise ValueError(
            f"Cannot run statistics-only for {result_dir}: missing or incompatible "
            "qunoc_circuit.pkl / exp_result.json compilation metadata. "
            "Use the original package/program/routing settings or recompile explicitly."
        )

    legacy_metadata_trusted = False
    if reusable_circuit and not recompile:
        legacy_metadata_trusted = legacy_reuse or cached.get("legacy_compilation_metadata_trusted", False)
        if legacy_reuse:
            logging.warning(
                "Re-estimating legacy circuit %s using matching path/program/routing metadata; "
                "old metadata has no topology fingerprint, so package topology must be unchanged.",
                circuit_path,
            )
        circuit = Circuit.from_pickle(circuit_path)
        logic_gate_num = cached["logic_gate_num"]
        metrics = Router.reestimate_compiled_circuit(package, circuit, logic_gate_num)
        statistics_source = "saved_circuit"
    else:
        program = get_program(quantum_program, data_qubit_num, **program_options)
        router = Router(
            package,
            program,
            route_strategy=str(routing_options["route_strategy"]),
            result_dir=result_dir,
            block_strategy=str(routing_options["block_strategy"]),
            move_min_gate_per_migrated_qubit=float(
                routing_options["move_min_gate_per_migrated_qubit"]
            ),
            move_cost_ratio=float(routing_options["move_cost_ratio"]),
            move_density_divisor=float(routing_options["move_density_divisor"]),
            ghz_source_strategy=str(routing_options["ghz_source_strategy"]),
            ghz_pairing_strategy=str(routing_options["ghz_pairing_strategy"]),
            tele_platform_strategy=str(routing_options["tele_platform_strategy"]),
            gate_select_strategy=str(routing_options["gate_select_strategy"]),
            gate_order_strategy=str(routing_options["gate_order_strategy"]),
        )
        circuit = router.route_program_on_package(
            mapping_strategy=str(routing_options["mapping_strategy"])
        )
        circuit.to_pickle(circuit_path)
        circuit_identity = _compiled_circuit_identity(circuit_path)
        logic_gate_num = program.get_gate_depend_graph().number_of_nodes() - 1
        details = router.statistic_route_details()
        metrics = {
            **details,
            "circuit_duration": router.estimate_qunoc_circuit_duration(details["eqst_batches"]),
            "gate_fidelity": router.estimate_qunoc_logic_gate_fide(details),
        }
        statistics_source = "new_compilation"

    exp_result = {
        "quantum_program": quantum_program,
        "data_qubit_num": data_qubit_num,
        "logic_gate_num": logic_gate_num,
        **{key: value for key, value in metrics.items() if key != "eqst_batches"},
        "signature": signature,
        "compilation_signature": compilation_signature,
        "compiled_circuit_identity": circuit_identity,
        "statistics_source": statistics_source,
        "legacy_compilation_metadata_trusted": legacy_metadata_trusted,
        "layout_package_path": package_path,
        "routing_options": routing_options,
        "program_options": program_options,
    }

    temporary = cache_path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as file:
        json.dump(exp_result, file, ensure_ascii=False, indent=2)
    os.replace(temporary, cache_path)

    return exp_result


def _run_experiment(common_defaults: dict[str, Any], spec: ExperimentSpec) -> None:
    os.makedirs(spec.exp_dir, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(spec.exp_dir, f"{spec.log_filename}.txt"),
        filemode="w",
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
        encoding="utf-8",
        force=True,
    )

    logging.info("Experiment: %s", spec.name)
    logging.info("Defaults: %s", spec.defaults)
    logging.info("Scalability: %s", spec.scalability)

    run_defaults = _deep_update(common_defaults, spec.defaults)
    scalability = copy.deepcopy(spec.scalability)
    quantum_programs = [
        str(value)
        for value in scalability.pop(
            "quantum_program_to_test",
            run_defaults.get("quantum_programs", []),
        )
    ]
    if not quantum_programs:
        raise ValueError(
            f"Routing experiment {spec.name} must define at least one quantum program."
        )
    axis_names, axis_values = _build_sweep_axes(scalability)
    if not axis_names:
        logging.info("Skip %s because no sweep axes were provided.", spec.name)
        return

    output_path = os.path.join(spec.exp_dir, spec.output_csv)
    # CSV is a snapshot of this sweep. Rebuild even when reusing per-case JSON,
    # so old-schema rows and results from another EQST model cannot be mixed in.
    with open(output_path, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [AXIS_LABELS.get(name, name) for name in axis_names]
            + [
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
            + EQST_CSV_FIELDS
        )

    for values in product(*axis_values):
        run_config = copy.deepcopy(run_defaults)
        for target_key, value in values:
            if target_key == "package_x_size":
                run_config.pop("package_size", None)
            _set_nested(run_config, target_key, value)

        for quantum_program in quantum_programs:
            exp_result = _route_once(
                common_defaults,
                spec,
                run_config,
                quantum_program,
            )

            with open(output_path, mode="a", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow(
                    [
                        *(_format_value(value) for _, value in values),
                        exp_result["quantum_program"],
                        exp_result["data_qubit_num"],
                        exp_result["logic_gate_num"],
                        exp_result["circuit_duration"],
                        exp_result["gate_fidelity"],
                        exp_result["intra_gate"],
                        exp_result["inter_gate"],
                        exp_result["qst_dist"],
                        exp_result["meas_num"],
                        exp_result["total_fide"],
                        exp_result["qst_effective_dist"],
                        exp_result["qst_neglog_fidelity"],
                        exp_result["total_neglog_fidelity"],
                        exp_result["eqst_model"],
                    ]
                )

            logging.info(
                "Finished %s -> duration=%s, fidelity=%s",
                _build_result_dir_name(
                    spec.name,
                    run_config,
                    exp_result["quantum_program"],
                ),
                exp_result["circuit_duration"],
                exp_result["gate_fidelity"],
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified routing scalability test runner.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).with_name("test_routing_scalability.yaml")),
        help="Path to a JSON/YAML config file.",
    )
    _add_eqst_arguments(parser)
    execution = parser.add_mutually_exclusive_group()
    execution.add_argument(
        "--statistics-only", action="store_true",
        help="Re-estimate saved circuits only; never regenerate programs or compile routing.",
    )
    execution.add_argument(
        "--recompile", action="store_true",
        help="Force new routing compilation instead of reusing a compatible saved circuit.",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_file():
        fallback = Path(__file__).with_name(config_path.name)
        if fallback.is_file():
            config_path = fallback

    cfg = load_experiment_config(config_path if config_path.is_file() else None)
    if not cfg.experiments:
        raise ValueError("No experiments were defined in the config file.")

    eqst_overrides = _eqst_cli_overrides(args)
    for spec in cfg.experiments.values():
        # Explicit CLI settings take precedence over both default scopes.
        if eqst_overrides:
            spec.defaults = _deep_update(spec.defaults, {"eqst": eqst_overrides})
        if args.statistics_only or args.recompile:
            spec.defaults["statistics_only"] = args.statistics_only
            spec.defaults["recompile"] = args.recompile
        _run_experiment(cfg.defaults, spec)


if __name__ == "__main__":
    main()