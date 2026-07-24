from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path
from typing import Any


EVALUATE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = EVALUATE_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baseline.MECH.MECHBenchmarks import MECH_experiments
from baseline.utils.utils import get_mech_circuit_duration
from qunoc.element.package import FabricateModel


ALGORITHMS = ["vqe", "qft", "qaoa", "bv"]
CHIPLET_LENGTH = 10
DEFAULT_INTER_SPARSITY = 3
DEFAULT_INTER_FIDE = 0.983
SENSATE_MAX_CAPACITY = 100
SENSATE_MAX_LENGTH = 80
BASE_OUT = EVALUATE_ROOT / "result" / "mechbaseline"

Case = dict[str, Any]

cache: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = {}


def data_qubit_num(package_size: tuple[int, int]) -> int:
    node_num = package_size[0] * package_size[1]
    return int(100 * 2**math.log(node_num / 1600, 5))


def divisor_at_most(n: int, limit: int) -> int:
    for divisor in range(min(n, limit), 0, -1):
        if n % divisor == 0:
            return divisor
    return 1


def exact_mech_shape(
    package_size: tuple[int, int], preferred_chiplet: int = CHIPLET_LENGTH
) -> tuple[tuple[int, int], tuple[int, int]]:
    x, y = package_size
    chiplet_x = divisor_at_most(x, preferred_chiplet)
    chiplet_y = divisor_at_most(y, preferred_chiplet)
    return (x // chiplet_x, y // chiplet_y), (chiplet_x, chiplet_y)


def _available_cross_link_counts(
    geometry: str, chiplet_size: tuple[int, int]
) -> tuple[int, int]:
    chiplet_x, chiplet_y = chiplet_size
    if geometry == "square":
        return chiplet_y, chiplet_x
    if geometry == "hexagon":
        return chiplet_y, len(range((chiplet_y + 1) % 2, chiplet_x, 2))
    if geometry == "heavy_square":
        return len(range(0, chiplet_y, 2)), len(range(0, chiplet_x, 2))
    if geometry == "heavy_hexagon":
        return len(range(0, chiplet_y, 2)), len(range(2, chiplet_x, 4))
    raise ValueError(f"Unsupported MECH geometry: {geometry}")


def sensate_spacing_to_mech_links(
    chiplet_size: tuple[int, int],
    sensate_spacing: int,
    geometry: str = "square",
) -> dict[str, Any]:
    if sensate_spacing <= 0:
        raise ValueError("Sensate inter_sparsity must be positive.")

    chiplet_x, chiplet_y = chiplet_size
    requested_row_links = math.ceil(chiplet_y / sensate_spacing)
    requested_col_links = math.ceil(chiplet_x / sensate_spacing)
    available_row_links, available_col_links = _available_cross_link_counts(
        geometry, chiplet_size
    )
    mech_row_links = min(requested_row_links, available_row_links)
    mech_col_links = min(requested_col_links, available_col_links)
    mech_link_count = min(mech_row_links, mech_col_links)
    return {
        "method": "link_count_only",
        "exact_topology_match": False,
        "sensate_inter_sparsity": sensate_spacing,
        "requested_row_link_count": requested_row_links,
        "requested_col_link_count": requested_col_links,
        "available_mech_row_link_count": available_row_links,
        "available_mech_col_link_count": available_col_links,
        "mech_row_link_count": mech_row_links,
        "mech_col_link_count": mech_col_links,
        "mech_link_count": mech_link_count,
        "uses_single_count_for_both_directions": True,
        "was_capped_by_geometry": (
            mech_row_links != requested_row_links
            or mech_col_links != requested_col_links
        ),
    }


def sensate_spacing_case_fields(
    chiplet_size: tuple[int, int],
    sensate_spacing: int = DEFAULT_INTER_SPARSITY,
    geometry: str = "square",
) -> Case:
    conversion = sensate_spacing_to_mech_links(
        chiplet_size, sensate_spacing, geometry
    )
    return {
        "inter_sparsity": sensate_spacing,
        "cross_link_sparsity": conversion["mech_link_count"],
    }


def comparison_fab_model(
    geometry: str = "square",
    inter_sparsity: int = DEFAULT_INTER_SPARSITY,
    inter_fide: float = DEFAULT_INTER_FIDE,
) -> FabricateModel:
    return FabricateModel(
        max_capacity=SENSATE_MAX_CAPACITY,
        max_length=SENSATE_MAX_LENGTH,
        compute_geometry=geometry,
        inter_sparsity=inter_sparsity,
        inter_fide=inter_fide,
    )


def get_field(row: dict[str, Any], *names: str, default: Any = 0) -> Any:
    for name in names:
        if name in row:
            return row[name]
    return default


def aligned_logic_gate_fidelity(
    exp_result: dict[str, Any], fab_model: FabricateModel
) -> float:
    logic_gate_num = exp_result["logic_gate_num"]
    if logic_gate_num <= 0:
        raise ValueError("Cannot normalize fidelity for a circuit without logic gates.")
    average_intra_num = exp_result["on_chip"] / logic_gate_num
    average_inter_num = exp_result["cross_chip"] / logic_gate_num
    average_qst_dist = exp_result.get("eqst_dist", 0) / logic_gate_num
    return (
        fab_model.get_intra_fide(fab_model.max_length) ** average_intra_num
        * fab_model.inter_fide**average_inter_num
        * fab_model.enhanced_fide**average_qst_dist
    )


def run_mech(
    structure: str,
    package_size: tuple[int, int],
    chiplet_array_dim: tuple[int, int],
    chiplet_size: tuple[int, int],
    algorithms: list[str],
    cross_link_sparsity: int,
    data_qubits: int,
) -> dict[str, dict[str, Any]]:
    key = (
        structure,
        package_size,
        chiplet_array_dim,
        chiplet_size,
        tuple(algorithms),
        cross_link_sparsity,
        data_qubits,
    )
    if key not in cache:
        result, _ = MECH_experiments(
            structure=structure,
            chiplet_array_dim=chiplet_array_dim,
            chiplet_size=chiplet_size,
            benchmarks=algorithms,
            data_qubit_num=data_qubits,
            cross_link_sparsity=cross_link_sparsity,
        )
        cache[key] = result
    return cache[key]


def write_rows(
    exp_name: str,
    headers: list[str],
    cases: list[Case],
) -> None:
    out_dir = Path(BASE_OUT) / exp_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "output.csv"

    with out_csv.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(
            ["Baseline"]
            + headers
            + [
                "Algorithm",
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
        )

        for case_index, case in enumerate(cases, start=1):
            print(
                f"[{exp_name}] running case {case_index}/{len(cases)}: "
                f"axis={case['axis_values']}",
                flush=True,
            )
            result = run_mech(
                case["structure"],
                case["package_size"],
                case["chiplet_array_dim"],
                case["chiplet_size"],
                ALGORITHMS,
                case["cross_link_sparsity"],
                case["data_qubits"],
            )

            raw_path = out_dir / f"{case['raw_name']}.json"
            with raw_path.open("w", encoding="utf-8") as raw_file:
                json.dump(result, raw_file, ensure_ascii=False, indent=2)

            model = comparison_fab_model(
                geometry=case.get("geometry", case["structure"]),
                inter_sparsity=case.get(
                    "inter_sparsity", DEFAULT_INTER_SPARSITY
                ),
                inter_fide=case.get("inter_fide", DEFAULT_INTER_FIDE),
            )

            for algorithm in ALGORITHMS:
                algorithm_result = result[algorithm]
                writer.writerow(
                    [
                        "mech",
                        *case["axis_values"],
                        algorithm,
                        case["data_qubits"],
                        get_field(algorithm_result, "logic_gate_num", default=""),
                        get_mech_circuit_duration(algorithm_result),
                        aligned_logic_gate_fidelity(algorithm_result, model),
                        get_field(algorithm_result, "on_chip", "on-chip"),
                        get_field(algorithm_result, "cross_chip", "cross-chip"),
                        get_field(algorithm_result, "eqst_dist", default=0),
                        get_field(algorithm_result, "meas_num"),
                        "",
                    ]
                )
            output_file.flush()

    print(out_csv, flush=True)
