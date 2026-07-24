# Legacy monolithic MECH runner. Prefer:
#   python evaluate/run_mech_routing.py --experiment all
import csv
import json
import math
import os

from baseline.MECH.MECHBenchmarks import MECH_experiments
from baseline.mech_utils import get_mech_circuit_duration, get_mech_logic_gate_fide
from qunoc.element.package import FabricateModel


ALGORITHMS = ["vqe", "qft", "qaoa", "bv"]
CHIPLET_LENGTH = 10
BASE_OUT = "result/routing_scalability/mech_baseline"

cache = {}


def data_qubit_num(package_size):
    node_num = package_size[0] * package_size[1]
    return int(100 * 2**math.log(node_num / 1600, 5))


def divisor_at_most(n, limit):
    for d in range(min(n, limit), 0, -1):
        if n % d == 0:
            return d
    return 1


def exact_mech_shape(package_size, preferred_chiplet=10):
    x, y = package_size
    cx = divisor_at_most(x, preferred_chiplet)
    cy = divisor_at_most(y, preferred_chiplet)
    return (x // cx, y // cy), (cx, cy)


def fab_model(chiplet_size, geometry="square", inter_sparsity=3, inter_fide=0.983):
    capacity = chiplet_size[0] * chiplet_size[1]
    return FabricateModel(
        max_capacity=capacity,
        max_length=int(0.8 * capacity),
        compute_geometry=geometry,
        inter_sparsity=inter_sparsity,
        inter_fide=inter_fide,
    )


def get_field(row, *names, default=0):
    for name in names:
        if name in row:
            return row[name]
    return default


def run_mech(
    structure,
    package_size,
    chiplet_array_dim,
    chiplet_size,
    algorithms,
    cross_link_sparsity,
    data_qubits,
):
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


def write_rows(exp_name, headers, cases):
    out_dir = os.path.join(BASE_OUT, exp_name)
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "output.csv")

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
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

        for case in cases:
            result = run_mech(
                case["structure"],
                case["package_size"],
                case["chiplet_array_dim"],
                case["chiplet_size"],
                ALGORITHMS,
                case["cross_link_sparsity"],
                case["data_qubits"],
            )

            raw_path = os.path.join(out_dir, case["raw_name"] + ".json")
            with open(raw_path, "w", encoding="utf-8") as jf:
                json.dump(result, jf, ensure_ascii=False, indent=2)

            fm = fab_model(
                case["chiplet_size"],
                geometry=case.get("geometry", case["structure"]),
                inter_sparsity=case.get("inter_sparsity", 3),
                inter_fide=case.get("inter_fide", 0.983),
            )

            for algorithm in ALGORITHMS:
                r = result[algorithm]
                writer.writerow(
                    [
                        "mech",
                        *case["axis_values"],
                        algorithm,
                        case["data_qubits"],
                        get_field(r, "logic_gate_num", default=""),
                        get_mech_circuit_duration(r),
                        get_mech_logic_gate_fide(r, case["chiplet_size"], fm),
                        get_field(r, "on_chip", "on-chip"),
                        get_field(r, "cross_chip", "cross-chip"),
                        get_field(r, "eqst_dist", default=0),
                        get_field(r, "meas_num"),
                        "",
                    ]
                )

    print(out_csv)


package_size_cases = []
for size in [40, 60, 80, 100, 120]:
    package_size = (size, size)
    chiplet_array_dim = (size // CHIPLET_LENGTH, size // CHIPLET_LENGTH)
    chiplet_size = (CHIPLET_LENGTH, CHIPLET_LENGTH)
    package_size_cases.append(
        {
            "structure": "square",
            "package_size": package_size,
            "chiplet_array_dim": chiplet_array_dim,
            "chiplet_size": chiplet_size,
            "cross_link_sparsity": 1,
            "data_qubits": data_qubit_num(package_size),
            "axis_values": [size],
            "raw_name": f"{size}_mech",
        }
    )

write_rows("package_size", ["PackageSize"], package_size_cases)

package_shape_cases = []
for x in [10, 20, 40, 50, 80, 100]:
    package_size = (x, 10000 // x)
    chiplet_array_dim, chiplet_size = exact_mech_shape(package_size, CHIPLET_LENGTH)
    package_shape_cases.append(
        {
            "structure": "square",
            "package_size": package_size,
            "chiplet_array_dim": chiplet_array_dim,
            "chiplet_size": chiplet_size,
            "cross_link_sparsity": 1,
            "data_qubits": data_qubit_num(package_size),
            "axis_values": [x],
            "raw_name": f"{package_size[0]}_{package_size[1]}_mech",
        }
    )

write_rows("package_shape", ["PackageXLength"], package_shape_cases)

geometry_cases = []
for geometry in ["square", "hexagon", "heavy_square", "heavy_hexagon"]:
    package_size = (100, 100)
    chiplet_array_dim, chiplet_size = exact_mech_shape(package_size, 10)
    if geometry == "heavy_hexagon":
        chiplet_array_dim, chiplet_size = (5, 5), (20, 20)
    geometry_cases.append(
        {
            "structure": geometry,
            "geometry": geometry,
            "package_size": package_size,
            "chiplet_array_dim": chiplet_array_dim,
            "chiplet_size": chiplet_size,
            "cross_link_sparsity": 1,
            "data_qubits": data_qubit_num(package_size),
            "axis_values": [geometry],
            "raw_name": f"{geometry}_mech",
        }
    )

write_rows("geometry", ["PackageGeometry"], geometry_cases)

inter_fidelity_cases = []
for inter_fide in [0.96, 0.965, 0.97, 0.975, 0.98, 0.985, 0.99]:
    package_size = (100, 100)
    chiplet_array_dim = (10, 10)
    chiplet_size = (10, 10)
    inter_fidelity_cases.append(
        {
            "structure": "square",
            "package_size": package_size,
            "chiplet_array_dim": chiplet_array_dim,
            "chiplet_size": chiplet_size,
            "cross_link_sparsity": 1,
            "inter_fide": inter_fide,
            "data_qubits": data_qubit_num(package_size),
            "axis_values": [inter_fide],
            "raw_name": f"{inter_fide}_mech",
        }
    )

write_rows("inter_fidelity", ["InterFidelity"], inter_fidelity_cases)

chiplet_sparsity_cases = []
for sparsity in [1, 3, 5, 7]:
    package_size = (100, 100)
    chiplet_array_dim = (10, 10)
    chiplet_size = (10, 10)
    chiplet_sparsity_cases.append(
        {
            "structure": "square",
            "package_size": package_size,
            "chiplet_array_dim": chiplet_array_dim,
            "chiplet_size": chiplet_size,
            "cross_link_sparsity": sparsity,
            "inter_sparsity": sparsity,
            "data_qubits": data_qubit_num(package_size),
            "axis_values": [sparsity],
            "raw_name": f"{sparsity}_mech",
        }
    )

write_rows("chiplet_sparsity", ["ChipletSparsity"], chiplet_sparsity_cases)
