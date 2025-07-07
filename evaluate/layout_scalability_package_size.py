import os
exp_dir = "./result/layout_scalability/package_size"
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
        writer.writerow(["PackageSize", "DesignStrategy", "Score", "Fidelity"])

from qunoc.element.package import *
from qunoc.layout.env import QuNoCEnv
from qunoc.layout.agent import LayoutAgent

package_length_to_test = range(40, 200, 20)
fab_model = FabricateModel()

for design_strategy in ["greedy", "rl"]:
    for package_length in package_length_to_test:
        logging.info(f"Testing package length: {package_length}, design strategy: {design_strategy}")
        result_dir = os.path.join(exp_dir, f"{package_length}_{package_length}_{design_strategy}")
        os.makedirs(result_dir, exist_ok=True)
        package_size = (package_length, package_length)

        if design_strategy == "baseline":
            package = Package(package_size, fab_model, design_strategy="rl")
            package.save(os.path.join(result_dir, "package_layout.pkl"))
        elif design_strategy == "greedy":
            try:
                package = Package(package_size, fab_model, design_strategy="greedy")
            except:
                package.save_design_history()
                raise
            package.save(os.path.join(result_dir, "package_layout.pkl"))
        elif design_strategy == "rl":
            env = QuNoCEnv(package_size, fab_model, patience=5)
            agent = LayoutAgent(env, result_dir=result_dir)
            agent.train(total_timesteps=100000)
            package = agent.design()
            package.save(os.path.join(result_dir, "package_layout.pkl"))

        score = package.estimate_score()
        fidelity = package.estimate_transfer()[0]
        with open(output_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([package_length, design_strategy, score, fidelity])
