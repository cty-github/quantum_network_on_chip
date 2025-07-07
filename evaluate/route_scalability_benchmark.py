import os
exp_dir = "./result/route_scalability/benchmark"
os.makedirs(exp_dir, exist_ok=True)

import sys
sys.path.append('..')

import logging
logging.basicConfig(
    filename=os.path.join(exp_dir, "log.txt"),
    filemode='w',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    encoding='utf-8'
)

import csv
output_path = os.path.join(exp_dir, "output.csv")
if not os.path.exists(output_path):
    with open(output_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Program", "DataQubitNum", "CircuitDepth", "GateFidelity"])

from baseline.MECH.MECHBenchmarks import MECH_experiments
from baseline.utils.utils import get_mech_circuit_depth, get_mech_gate_fidelity
from qunoc.element.benchmark import get_program
from qunoc.element.package import *
from qunoc.routing.router import *

package_length_to_test = range(40, 200, 20)
fab_model = FabricateModel()
op_config = OperationConfig()

for algorithm in ["vqe", "qft", "qaoa", "bv"]:
    for package_length in package_length_to_test:
        for baseline in ["mech", "qunoc"]:
            logging.info(f"Testing package length: {package_length}, program: {algorithm}, baseline: {baseline}")
            result_dir = os.path.join(exp_dir, f"{package_length}_{package_length}_{algorithm}_{baseline}")
            os.makedirs(result_dir, exist_ok=True)
            package_size = (package_length, package_length)
            data_qubit_num = None
            if baseline == "mech":
                exp_result_dict = MECH_experiments(structure="square", chiplet_array_dim=(package_length//10, package_length//10), chiplet_size=(10, 10), benchmarks=[algorithm])
                exp_result = exp_result_dict[algorithm]
                data_qubit_num = exp_result["data_qubit_num"]
                circuit_depth = get_mech_circuit_depth(exp_result, op_config)
                gate_fidelity = get_mech_gate_fidelity(exp_result, (10, 10), fab_model)
            elif baseline == "qunoc":
                package = Package(pre_designed_package_path=os.path.join("./result/layout_scalability/package_size", f"{package_length}_{package_length}_greedy/package_layout.pkl"))
                if data_qubit_num:
                    program = get_program(algorithm, data_qubit_num)
                router = Router(package, program, result_dir)
                circuit = router.route_program_on_package()
                circuit_depth = router.estimate_depth()
                gate_fidelity = router.estimate_average_gate_fide()
