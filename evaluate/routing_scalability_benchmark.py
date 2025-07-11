import os
import sys
sys.path.append("..")
import logging
import csv

from qunoc.utils.utils import get_timestamp
from qunoc.element.benchmark import get_program
from qunoc.element.package import *
from qunoc.routing.router import *
from baseline.MECH.MECHBenchmarks import MECH_experiments
from baseline.utils.utils import get_mech_data_qubit_num, get_mech_circuit_depth, get_mech_gate_fidelity

exp_dir = "./result/routing_scalability/benchmark"
os.makedirs(exp_dir, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(exp_dir, f"log_{get_timestamp()}.txt"),
    filemode='w',
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    encoding="utf-8"
)

output_path = os.path.join(exp_dir, "output.csv")
if not os.path.exists(output_path):
    with open(output_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Algorithm", "#DataQubit", "Baseline", "CircuitDepth", "GateFidelity"])

baseline_to_test = ["mech", "qunoc"]
algorithm_to_test = ["vqe", "qft", "qaoa", "bv"]
package_length_to_test = list(range(40, 220, 20))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--baseline", type=str, default=None, help="Baseline: mech, qunoc")
    parser.add_argument("--algorithm", type=str, default=None, help="Algorithm: vqe, qft, qaoa, bv")
    parser.add_argument("--length", type=int, default=None, help="Package length: 40, 60, 80, ...")
    args = parser.parse_args()
    if args.baseline:
        baseline_to_test = [args.baseline]
    if args.algorithm:
        algorithm_to_test = [args.algorithm]
    if args.length:
        package_length_to_test = list(range(args.length, 220, 20))

logging.info(f"Baselines to test: {baseline_to_test}")
logging.info(f"Algorithms to test: {algorithm_to_test}")
logging.info(f"Package lengths to test: {package_length_to_test}")
for baseline, algorithm, package_length in product(baseline_to_test, algorithm_to_test, package_length_to_test):
    logging.info(f"Testing package length: {package_length}, algorithm: {algorithm}, baseline: {baseline}")
    result_dir = os.path.join(exp_dir, f"{package_length}_{package_length}_{algorithm}_{baseline}")
    os.makedirs(result_dir, exist_ok=True)
    package_size = (package_length, package_length)
    data_qubit_num = package_size[0] * package_size[1] // 100

    if baseline == "mech":
        exp_result_dict = MECH_experiments(structure="square", chiplet_array_dim=(package_length//10, package_length//10), chiplet_size=(10, 10), benchmarks=[algorithm], data_qubit_num=data_qubit_num)
        exp_result = exp_result_dict[algorithm]
        import json
        with open(os.path.join(result_dir, "exp_result.json"), 'w') as json_file:
            json.dump(exp_result, json_file)
        circuit_depth = get_mech_circuit_depth(exp_result)
        gate_fidelity = get_mech_gate_fidelity(exp_result, (10, 10))
    elif baseline == "qunoc":
        package = Package(pre_designed_package_path=os.path.join("./result/layout_scalability/package_size", f"{package_length}_{package_length}_greedy/package_layout.pkl"))
        program = get_program(algorithm, data_qubit_num)
        router = Router(package, program, result_dir)
        circuit = router.route_program_on_package()
        intra_gate_num, inter_gate_num, qst_dist, total_fide = router.statistic_op_num_fide()
        circuit_depth = router.estimate_depth()
        gate_fidelity = router.estimate_average_gate_fide()
    
    with open(output_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([algorithm, data_qubit_num, baseline, circuit_depth, gate_fidelity])
    logging.info(f"Tested circuit depth: {circuit_depth}, gate fidelity: {gate_fidelity}")
