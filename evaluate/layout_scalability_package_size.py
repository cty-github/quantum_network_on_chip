import os
exp_dir = "result/layout_scalability/package_size/"
os.makedirs(exp_dir, exist_ok=True)

import sys
sys.path.append('..')

import logging
logging.basicConfig(level=logging.WARN, filename=f"{exp_dir}output.log",)

import csv
csv_path = f"{exp_dir}transfer_fidelity.csv"
with open(csv_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["PackageSize", "Baseline", "Greedy", "RL"])

# from qunoc.utils.utils import get_timestamp
from qunoc.element.package import *
from qunoc.layout.env import QuNoCEnv
from qunoc.layout.agent import LayoutAgent

package_length_to_test = range(50, 500, 100)
fab_model = FabricateModel()

for package_length in package_length_to_test:
    result_dir = f"{exp_dir}{package_length}/"
    os.makedirs(result_dir, exist_ok=True)
    package_size = (package_length, package_length)

    package_baseline = Package(package_size, fab_model, design_strategy="rl")
    package_greedy = Package(package_size, fab_model, design_strategy="greedy", result_dir=result_dir)

    env = QuNoCEnv(package_size, fab_model)
    agent = LayoutAgent(env, result_dir=result_dir)
    agent.train()
    package_rl = agent.design()

    package_list = [package_baseline, package_greedy, package_rl]
    fidelity_list = [package.estimate_transfer() for package in package_list]

    with open(csv_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([package_length, *fidelity_list])
