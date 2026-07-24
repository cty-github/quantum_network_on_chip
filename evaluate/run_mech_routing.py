import argparse

try:
    from .mech_routing_common import (
        CHIPLET_LENGTH,
        Case,
        data_qubit_num,
        exact_mech_shape,
        sensate_spacing_case_fields,
        write_rows,
    )
except ImportError:
    from mech_routing_common import (
        CHIPLET_LENGTH,
        Case,
        data_qubit_num,
        exact_mech_shape,
        sensate_spacing_case_fields,
        write_rows,
    )


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
