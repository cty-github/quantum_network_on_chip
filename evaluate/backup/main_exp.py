# Legacy summary script. Prefer the four-step workflow in evaluate/evaluate.md
# and scripted figure generation through evaluate/plot_result_figures.py.
import os
import sys
import csv
import json
import math
from itertools import product

sys.path.append("..")
from baseline.mech_utils import get_mech_highway_proportion
from baseline.mech_utils import duration_modify_const
from qunoc.element.chiplet import FabricateModel
from qunoc.element.package import Package
from qunoc.routing.circuit import Circuit

result_dir = "./result/routing_scalability/overall"

baseline_to_test = ["mech", "qunoc"]
algorithm_to_test = ["vqe", "qft", "qaoa"]
package_length_to_test = list(range(40, 140, 20))

output_csv_path = os.path.join(result_dir, "output.csv")
output_file = open("./main_exp.csv", mode="w", newline="")
writer = csv.writer(output_file)
writer.writerow(
    [
        "Algorithm",
        "#DataQubit",
        "QuNoCProportion",
        "MECHGateFidelity",
        "QuNoCGateFidelity",
        "Improvement",
        "MECHCircuitDuration",
        "QuNoCCircuitDuration",
        "Speedup",
    ]
)

for algorithm, package_length in product(algorithm_to_test, package_length_to_test):
    node_num = package_length * package_length
    data_qubit_num = int(100 * 2 ** math.log(node_num / 1600, 5))
    fab_model = FabricateModel()

    mech_duration = mech_fidelity = qunoc_duration = qunoc_fidelity = None
    with open(output_csv_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if (row["Algorithm"] == algorithm and
                int(row["#DataQubit"]) == data_qubit_num and
                row["Baseline"] == "mech"):
                mech_duration = float(row["CircuitDuration"])
                mech_fidelity = float(row["GateFidelity"])
            elif (row["Algorithm"] == algorithm and
                  int(row["#DataQubit"]) == data_qubit_num and
                  row["Baseline"] == "qunoc"):
                qunoc_duration = float(row["CircuitDuration"])
                qunoc_fidelity = float(row["GateFidelity"])

    if None in (mech_duration, mech_fidelity, qunoc_duration, qunoc_fidelity):
        print(f"Skipping {algorithm} @ {data_qubit_num} due to missing data.")
        continue



    # chiplet_array_dim = (package_length // 10, package_length // 10)
    # chiplet_size = (10, 10)
    # cross_link_sparsity = chiplet_size[0] // fab_model.inter_sparsity
    # mech_proportion = get_mech_highway_proportion(
    #     structure="square",
    #     chiplet_array_dim=chiplet_array_dim,
    #     chiplet_size=chiplet_size,
    #     cross_link_sparsity=cross_link_sparsity,
    # )

    mech_duration = mech_duration / 1000000 * duration_modify_const


    package = Package.from_pickle(filepath=os.path.join("./result/layout_scalability/package_size", f"{package_length}_{package_length}_greedy/package.pkl"))
    qunoc_proportion = len(package.transfer_nodes) / (package_length ** 2)

    circuit = Circuit.from_pickle(os.path.join(result_dir, f"{package_length}_{package_length}_{algorithm}_qunoc/qunoc_circuit.pkl"))
    qunoc_duration = circuit.estimate_duration(package) / 1000000

    # with open(os.path.join(qunoc_result_dir, "exp_result.json"), "r") as f:
    #     qunoc_exp_result = json.load(f)
    # qunoc_gate_num = qunoc_exp_result.get("logic_gate_num")
    # qunoc_intra_num = qunoc_exp_result.get("intra_gate") / qunoc_gate_num
    # qunoc_inter_num = qunoc_exp_result.get("inter_gate") / qunoc_gate_num
    # qunoc_qst_dist = qunoc_exp_result.get("qst_dist") / qunoc_gate_num
    # qunoc_fidelity = (fab_model.get_intra_fide(chiplet_size[0] * chiplet_size[1]) ** qunoc_intra_num) * (fab_model.inter_fide ** qunoc_inter_num) * (fab_model.enhanced_fide ** qunoc_qst_dist)
    
    # package = Package.from_pickle(
    #     filepath=os.path.join(
    #         "./result/layout_scalability/package_size",
    #         f"{package_length}_{package_length}_greedy/package.pkl",
    #     )
    # )

    writer.writerow(
        [
            algorithm,
            data_qubit_num,
            qunoc_proportion, 
            mech_fidelity,
            qunoc_fidelity,
            qunoc_fidelity / mech_fidelity,
            mech_duration,
            qunoc_duration,
            mech_duration / qunoc_duration,
        ]
    )

output_file.close()