import os
import sys

sys.path.append("..")
import csv
import logging

from qunoc.element.package import Package, FabricateModel, product, np
from qunoc.layout.env import QuNoCEnv
from qunoc.layout.agent import LayoutAgent

exp_dir = "./result/layout_scalability/chiplet_sparsity"
os.makedirs(exp_dir, exist_ok=True)

design_strategy_to_test = ["baseline", "greedy", "rl"]
chiplet_sparsity_to_test = [1, 3, 5, 7]
log_filename = "log"
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--strategy",
        type=str,
        default=None,
        help="Design strategy: baseline, heuristic, greedy, or rl",
    )
    parser.add_argument(
        "--sparsity", type=int, default=None, help="Chiplet sparsity: 1, 3, 5, 7"
    )
    args = parser.parse_args()
    if args.strategy:
        design_strategy_to_test = [args.strategy]
        log_filename += f"_{args.strategy}"
    if args.sparsity:
        chiplet_sparsity_to_test = [args.sparsity]
        log_filename += f"_{args.sparsity}"

logging.basicConfig(
    filename=os.path.join(exp_dir, f"{log_filename}.txt"),
    filemode="w",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    encoding="utf-8",
)
logging.info(f"Design strategies to test: {design_strategy_to_test}")
logging.info(f"Chiplet sparsities to test: {chiplet_sparsity_to_test}")

output_path = os.path.join(exp_dir, "output.csv")
if not os.path.exists(output_path):
    with open(output_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["ChipletSparsity", "DesignStrategy", "Score", "Fidelity", "FideStd"]
        )

for design_strategy, chiplet_sparsity in product(
    design_strategy_to_test, chiplet_sparsity_to_test
):
    logging.info(
        f"Testing chiplet sparsity: {chiplet_sparsity}, design strategy: {design_strategy}"
    )
    result_dir = os.path.join(
        exp_dir, f"{chiplet_sparsity}_{design_strategy}"
    )
    os.makedirs(result_dir, exist_ok=True)
    package_size = (100, 100)
    fab_model = FabricateModel(inter_sparsity=chiplet_sparsity)

    if os.path.exists(os.path.join(result_dir, "package.pkl")):
        package = Package.from_pickle(os.path.join(result_dir, "package.pkl"))
    else:
        if design_strategy == "baseline":
            package = Package(
                package_size=package_size,
                fab_model=fab_model,
                design_scheme="bottom_up",
                design_strategy="manual",
                bottom_up_init_type=np.ones(package_size, dtype=np.int8),
                result_dir=result_dir,
            )
        elif design_strategy == "strip":
            package = Package(
                package_size=package_size,
                fab_model=fab_model, 
                design_strategy="strip", 
                result_dir=result_dir
            )
        elif design_strategy == "grid":
            package = Package(
                package_size=package_size,
                fab_model=fab_model, 
                design_strategy="grid", 
                result_dir=result_dir
            )
        elif design_strategy == "radial":
            package = Package(
                package_size=package_size,
                fab_model=fab_model, 
                design_strategy="radial", 
                result_dir=result_dir
            )
        elif design_strategy == "heuristic":
            package = Package(
                package_size=package_size,
                fab_model=fab_model, 
                design_strategy="heuristic", 
                result_dir=result_dir
            )
        elif design_strategy == "greedy":
            package = Package(
                package_size=package_size,
                fab_model=fab_model, 
                design_strategy="greedy", 
                result_dir=result_dir
            )
        elif design_strategy == "rl":
            env = QuNoCEnv(
                package_size=package_size, 
                fab_model=fab_model, 
                patience=5, 
                result_dir=result_dir
            )
            if os.path.exists(os.path.join(result_dir, "layout_agent.pth")):
                agent = LayoutAgent(
                    env,
                    pre_trained_model_path=os.path.join(result_dir, "layout_agent.pth"),
                )
            else:
                agent = LayoutAgent(env, result_dir=result_dir)
                agent.train(total_timesteps=100 * package_size[0] * package_size[1])
            package = agent.design()
        package.to_pickle()
        package.save_design_history()
        try:
            fig, _ = package.draw_qubits(axis_off=False)
            fig.savefig(os.path.join(result_dir, "package.png"))
            logging.info("Saved package visualizing results")
        except Exception as e:
            logging.warning(f"Failed to draw qubits: {e}")

    score = package.estimate_score()
    fidelity, fide_std = package.estimate_transfer()
    with open(output_path, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([chiplet_sparsity, design_strategy, score, fidelity, fide_std])
    logging.info(f"Tested score: {score}, fidelity: {fidelity}, fide_std: {fide_std}")
