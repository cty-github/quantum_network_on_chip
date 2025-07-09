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

design_strategy_to_test = ["baseline", "greedy", "rl"]
package_length_to_test = list(range(40, 220, 20))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--strategy", type=str, default=None, help="Design strategy: baseline, greedy, or rl")
    parser.add_argument("--length", type=int, default=None, help="Package length: 40, 60, 80, ...")
    args = parser.parse_args()
    if args.strategy:
        design_strategy_to_test = [args.strategy]
    if args.length:
        package_length_to_test = list(range(args.length, 220, 20))

logging.info(f"Design strategies to test: {design_strategy_to_test}")
logging.info(f"Package lengths to test: {package_length_to_test}")
for design_strategy, package_length in product(design_strategy_to_test, package_length_to_test):
        logging.info(f"Testing package length: {package_length}, design strategy: {design_strategy}")
        result_dir = os.path.join(exp_dir, f"{package_length}_{package_length}_{design_strategy}")
        os.makedirs(result_dir, exist_ok=True)
        package_size = (package_length, package_length)

        if design_strategy == "baseline":
            package = Package(package_size, design_strategy="rl", result_dir=result_dir)
            package.save()
        elif design_strategy == "greedy":
            package = Package(package_size, design_strategy="greedy", result_dir=result_dir)
            package.save()
        elif design_strategy == "rl":
            env = QuNoCEnv(package_size, patience=5)
            agent = LayoutAgent(env, result_dir=result_dir)
            agent.train(total_timesteps=100000)
            package = agent.design()
            package.save()

        score = package.estimate_score()
        fidelity = package.estimate_transfer()[0]
        with open(output_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([package_length, design_strategy, score, fidelity])
        logging.info(f"Tested score: {score}, fidelity: {fidelity}")
