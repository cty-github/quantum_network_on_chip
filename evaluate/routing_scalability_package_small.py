import os
import sys

sys.path.append("..")
import csv
import json
import logging

import math
import pickle
from qunoc.element.benchmark import get_program
from qunoc.element.package import product, Package, FabricateModel
from qunoc.routing.router import Router
from baseline.MECH.MECHBenchmarks import MECH_experiments
from baseline.utils.utils import get_mech_circuit_duration, get_mech_logic_gate_fide

exp_dir = "./result/routing_scalability/overall"
os.makedirs(exp_dir, exist_ok=True)

baseline_to_test = ["mech", "qunoc"]
algorithm_to_test = ["vqe", "qft", "qaoa", "bv"]
package_length_to_test = [60, 80]
log_filename = "log_small"
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--baseline", type=str, default=None, help="Baseline: mech, qunoc"
    )
    parser.add_argument(
        "--algorithm", type=str, default=None, help="Algorithm: vqe, qft, qaoa, bv"
    )
    parser.add_argument(
        "--length", type=int, default=None, help="Package length: 40, 60, 80, ..."
    )
    args = parser.parse_args()
    if args.baseline:
        baseline_to_test = [args.baseline]
        log_filename += f"_{args.baseline}"
    if args.algorithm:
        algorithm_to_test = [args.algorithm]
        log_filename += f"_{args.algorithm}"
    if args.length:
        package_length_to_test = [args.length]
        log_filename += f"_{args.length}"

logging.basicConfig(
    filename=os.path.join(exp_dir, f"{log_filename}.txt"),
    filemode="w",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    encoding="utf-8",
)
logging.info(f"Baselines to test: {baseline_to_test}")
logging.info(f"Algorithms to test: {algorithm_to_test}")
logging.info(f"Package lengths to test: {package_length_to_test}")

output_path = os.path.join(exp_dir, "output.csv")
if not os.path.exists(output_path):
    with open(output_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "Algorithm",
                "#DataQubit",
                "#LogicGate",
                "Baseline",
                "CircuitDuration",
                "GateFidelity",
            ]
        )

for baseline, algorithm, package_length in product(
    baseline_to_test, algorithm_to_test, package_length_to_test
):
    logging.info(
        f"Testing package: {package_length}*{package_length}, algorithm: {algorithm}, baseline: {baseline}"
    )
    result_dir = os.path.join(
        exp_dir, f"{package_length}_{package_length}_{algorithm}_{baseline}"
    )
    os.makedirs(result_dir, exist_ok=True)
    # if os.path.exists(os.path.join(result_dir, "exp_result.json")):
    #     logging.info(
    #         f"Result already exists for {package_width}_{package_height}_{algorithm}_{baseline}, skipping..."
    #     )
    #     continue
    package_size = (package_length, package_length)
    node_num = math.prod(package_size)
    data_qubit_num = int(100 * 2 ** math.log(node_num / 1600, 5))

    if baseline == "mech":
        chiplet_array_dim = (package_length // 10, package_length // 10)
        chiplet_size = (10, 10)
        cross_link_sparsity = chiplet_size[0] // FabricateModel().inter_sparsity
        exp_result_dict, circuit_dict = MECH_experiments(
            structure="square",
            chiplet_array_dim=chiplet_array_dim,
            chiplet_size=chiplet_size,
            benchmarks=[algorithm],
            data_qubit_num=data_qubit_num,
            cross_link_sparsity=cross_link_sparsity,
        )
        exp_result = exp_result_dict[algorithm]
        circuit = circuit_dict[algorithm]
        with open(os.path.join(result_dir, "mech_circuit.pkl"), "wb") as f:
            pickle.dump(circuit, f)
        circuit_duration = get_mech_circuit_duration(exp_result)
        logic_gate_fide = get_mech_logic_gate_fide(exp_result, (10, 10))
    elif baseline == "qunoc":
        package = Package.from_pickle(
            filepath=os.path.join(
                "./result/layout_scalability/package_size",
                f"{package_length}_{package_length}_greedy/package.pkl",
            )
        )
        program = get_program(algorithm, data_qubit_num)
        router = Router(package, program, result_dir=result_dir)
        circuit = router.route_program_on_package()
        circuit.to_pickle(os.path.join(result_dir, "qunoc_circuit.pkl"))
        intra_gate_num, inter_gate_num, qst_dist, meas_num, total_fide = (
            router.statistic_route_results()
        )
        circuit_duration = router.estimate_qunoc_circuit_duration()
        logic_gate_fide = router.estimate_qunoc_logic_gate_fide()
        exp_result = {
            "data_qubit_num": data_qubit_num,
            "logic_gate_num": program.get_gate_depend_graph().number_of_nodes() - 1,
            "duration": circuit_duration,
            "fidelity": logic_gate_fide,
            "intra_gate": intra_gate_num,
            "inter_gate": inter_gate_num,
            "qst_dist": qst_dist,
            "meas_num": meas_num,
            "sparsity": FabricateModel().inter_sparsity,
        }

    with open(os.path.join(result_dir, "exp_result.json"), "w") as json_file:
        json.dump(exp_result, json_file)

    with open(output_path, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                algorithm,
                data_qubit_num,
                exp_result["logic_gate_num"],
                baseline,
                circuit_duration,
                logic_gate_fide,
            ]
        )
    logging.info(
        f"Tested circuit depth: {circuit_duration}, gate fidelity: {logic_gate_fide}"
    )
