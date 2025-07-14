import os
import sys
sys.path.append("..")
import logging
import csv

from qunoc.utils.utils import get_timestamp
from qunoc.element.package import *
from qunoc.layout.env import QuNoCEnv
from qunoc.layout.agent import LayoutAgent

exp_dir = "./result/layout_scalability/package_size"
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
        writer.writerow(["PackageSize", "DesignStrategy", "Score", "Fidelity"])

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
    elif design_strategy == "greedy":
        package = Package(package_size, design_strategy="greedy", result_dir=result_dir)
    elif design_strategy == "rl":
        env = QuNoCEnv(package_size, patience=5, result_dir=result_dir)
        if os.path.exists(os.path.join(result_dir, "layout_agent.pth")):
            agent = LayoutAgent(env, pre_trained_model_path=os.path.join(result_dir, "layout_agent.pth"))
        else:
            agent = LayoutAgent(env, result_dir=result_dir)
            agent.train(total_timesteps=100*package_size[0]*package_size[1])
        package = agent.design()
    
    score = package.estimate_score()
    fidelity = package.estimate_transfer()[0]
    with open(output_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([package_length, design_strategy, score, fidelity])
    logging.info(f"Tested score: {score}, fidelity: {fidelity}")

    package.save()
    package.save_design_history()
    try:
        fig, _ = package.draw_qubits(axis_off=False)
        fig.savefig(os.path.join(result_dir, "package.png"))
        logging.info(f"Saved package visualizing results")
    except Exception as e:
        logging.warning(f"Failed to draw qubits: {e}")
