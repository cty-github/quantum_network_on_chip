import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baseline.mech_utils import (
    CHIPLET_LENGTH,
    Case,
    aligned_logic_gate_fidelity,
    DEFAULT_INTER_FIDE,
    DEFAULT_INTER_SPARSITY,
    comparison_fab_model,
    data_qubit_num,
    exact_mech_shape,
    get_field,
    get_mech_circuit_duration,
    run_mech,
    sensate_spacing_case_fields,
)


ALGORITHMS = ["vqe", "qft", "qaoa", "bv"]
BASE_OUT = PROJECT_ROOT / "evaluate" / "result" / "mechbaseline"

EXPERIMENT_HEADERS = {
    "package_size": ["PackageSize"],
    "package_shape": ["PackageXLength"],
    "geometry": ["PackageGeometry"],
    "inter_fidelity": ["InterFidelity"],
    "chiplet_sparsity": ["ChipletSparsity"],
}


def build_package_size_cases() -> list[Case]:
    cases = []
    chiplet_size = (CHIPLET_LENGTH, CHIPLET_LENGTH)
    for size in [40, 60, 80, 100, 120]:
        package_size = (size, size)
        cases.append(
            {
                "structure": "square",
                "package_size": package_size,
                "chiplet_array_dim": (size // CHIPLET_LENGTH, size // CHIPLET_LENGTH),
                "chiplet_size": chiplet_size,
                "data_qubits": data_qubit_num(package_size),
                "axis_values": [size],
                "raw_name": f"{size}_mech",
                **sensate_spacing_case_fields(chiplet_size),
            }
        )
    return cases


def build_package_shape_cases() -> list[Case]:
    cases = []
    for x_length in [10, 20, 40, 50, 80, 100]:
        package_size = (x_length, 10000 // x_length)
        chiplet_array_dim, chiplet_size = exact_mech_shape(package_size, CHIPLET_LENGTH)
        cases.append(
            {
                "structure": "square",
                "package_size": package_size,
                "chiplet_array_dim": chiplet_array_dim,
                "chiplet_size": chiplet_size,
                "data_qubits": data_qubit_num(package_size),
                "axis_values": [x_length],
                "raw_name": f"{package_size[0]}_{package_size[1]}_mech",
                **sensate_spacing_case_fields(chiplet_size),
            }
        )
    return cases


def build_geometry_cases() -> list[Case]:
    cases = []
    for geometry in ["square", "hexagon", "heavy_square", "heavy_hexagon"]:
        package_size = (100, 100)
        chiplet_array_dim, chiplet_size = exact_mech_shape(package_size, CHIPLET_LENGTH)
        if geometry == "heavy_hexagon":
            chiplet_array_dim, chiplet_size = (5, 5), (20, 20)
        cases.append(
            {
                "structure": geometry,
                "geometry": geometry,
                "package_size": package_size,
                "chiplet_array_dim": chiplet_array_dim,
                "chiplet_size": chiplet_size,
                "data_qubits": data_qubit_num(package_size),
                "axis_values": [geometry],
                "raw_name": f"{geometry}_mech",
                **sensate_spacing_case_fields(chiplet_size, geometry=geometry),
            }
        )
    return cases


def build_inter_fidelity_cases() -> list[Case]:
    cases = []
    chiplet_size = (CHIPLET_LENGTH, CHIPLET_LENGTH)
    for inter_fide in [0.96, 0.965, 0.97, 0.975, 0.98, 0.985, 0.99]:
        package_size = (100, 100)
        cases.append(
            {
                "structure": "square",
                "package_size": package_size,
                "chiplet_array_dim": (10, 10),
                "chiplet_size": chiplet_size,
                "inter_fide": inter_fide,
                "data_qubits": data_qubit_num(package_size),
                "axis_values": [inter_fide],
                "raw_name": f"{inter_fide}_mech",
                **sensate_spacing_case_fields(chiplet_size),
            }
        )
    return cases


def build_chiplet_sparsity_cases() -> list[Case]:
    cases = []
    chiplet_size = (CHIPLET_LENGTH, CHIPLET_LENGTH)
    for sparsity in [1, 3, 5, 7]:
        package_size = (100, 100)
        cases.append(
            {
                "structure": "square",
                "package_size": package_size,
                "chiplet_array_dim": (10, 10),
                "chiplet_size": chiplet_size,
                "inter_sparsity": sparsity,
                "data_qubits": data_qubit_num(package_size),
                "axis_values": [sparsity],
                "raw_name": f"{sparsity}_mech",
                **sensate_spacing_case_fields(chiplet_size, sparsity),
            }
        )
    return cases


CASE_BUILDERS = {
    "package_size": build_package_size_cases,
    "package_shape": build_package_shape_cases,
    "geometry": build_geometry_cases,
    "inter_fidelity": build_inter_fidelity_cases,
    "chiplet_sparsity": build_chiplet_sparsity_cases,
}


def write_rows(
    exp_name: str,
    headers: list[str],
    cases: list[Case],
) -> None:
    out_dir = BASE_OUT / exp_name
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
                inter_sparsity=case.get("inter_sparsity", DEFAULT_INTER_SPARSITY),
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


def run_experiment(experiment: str) -> None:
    cases = CASE_BUILDERS[experiment]()
    write_rows(experiment, EXPERIMENT_HEADERS[experiment], cases)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MECH routing baseline sweeps.")
    parser.add_argument(
        "--experiment",
        choices=[*CASE_BUILDERS.keys(), "all"],
        default="all",
        help="MECH experiment to run.",
    )
    args = parser.parse_args()

    experiments = CASE_BUILDERS.keys() if args.experiment == "all" else [args.experiment]
    for experiment in experiments:
        run_experiment(experiment)


if __name__ == "__main__":
    main()
