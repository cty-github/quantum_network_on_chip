import os
import sys
sys.path.append('..')
import logging
import csv

from qunoc.utils.utils import get_timestamp
from qunoc.element.package import *
from qunoc.layout.env import QuNoCEnv
from qunoc.layout.agent import LayoutAgent

exp_dir = "./result/layout_scalability/package_shape"
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
        writer.writerow(["PackageXSize", "DesignStrategy", "Score", "Fidelity"])

design_strategy_to_test = ["baseline", "greedy", "rl"]
package_x_size_to_test = [10, 20, 40, 50, 80, 100]

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--strategy", type=str, default=None, help="Design strategy: baseline, greedy, or rl")
    parser.add_argument("--x_size", type=int, default=None, help="Package x size: 10, 20, 40, 50, 80, 100")
    args = parser.parse_args()
    if args.strategy:
        design_strategy_to_test = [args.strategy]
    if args.x_size:
        package_x_size_to_test = [args.length]

logging.info(f"Design strategies to test: {design_strategy_to_test}")
logging.info(f"Package x size to test: {package_x_size_to_test}")
for design_strategy, package_x_size in product(design_strategy_to_test, package_x_size_to_test):
        package_y_size = 10000 // package_x_size
        logging.info(f"Testing package shape: ({package_x_size}, {package_y_size}), design strategy: {design_strategy}")
        result_dir = os.path.join(exp_dir, f"{package_x_size}_{package_y_size}_{design_strategy}")
        os.makedirs(result_dir, exist_ok=True)
        package_size = (package_x_size, package_y_size)

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
            writer.writerow([package_x_size, design_strategy, score, fidelity])
        logging.info(f"Tested score: {score}, fidelity: {fidelity}")

        package.save()
        # package.save_design_history()
        try:
            fig, _ = package.draw_qubits(axis_off=False)
            fig.savefig(os.path.join(result_dir, "package.png"))
            logging.info(f"Saved package visualizing results")
        except Exception as e:
            logging.warning(f"Failed to draw qubits: {e}")
