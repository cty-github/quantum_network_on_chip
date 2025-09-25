import os
import sys

sys.path.append("..")
import csv
import logging

from qunoc.element.package import Package, product, np, FabricateModel
from qunoc.layout.env import QuNoCEnv
from qunoc.layout.agent import LayoutAgent

exp_dir = "./result/layout_scalability/geometry"
os.makedirs(exp_dir, exist_ok=True)

design_strategy_to_test = ["baseline", "greedy"]
package_geometry_to_test = ["square", "hexagon", "heavy_square", "heavy_hexagon"]

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
        "--geometry",
        type=str,
        default=None,
        help="Package geometry: square, hexagon, heavy_square, heavy_hexagon",
    )
    args = parser.parse_args()
    if args.strategy:
        design_strategy_to_test = [args.strategy]
        log_filename += f"_{args.strategy}"
    if args.geometry:
        package_geometry_to_test = [args.geometry]
        log_filename += f"_{args.geometry}"

logging.basicConfig(
    filename=os.path.join(exp_dir, f"{log_filename}.txt"),
    filemode="w",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    encoding="utf-8",
)
logging.info(f"Design strategies to test: {design_strategy_to_test}")
logging.info(f"Package geometries to test: {package_geometry_to_test}")

output_path = os.path.join(exp_dir, "output.csv")
if not os.path.exists(output_path):
    with open(output_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["PackageGeometry", "DesignStrategy", "Score", "Fidelity", "FideStd"]
        )

for design_strategy, package_geometry in product(
    design_strategy_to_test, package_geometry_to_test
):
    logging.info(
        f"Testing package geometry: {package_geometry}, design strategy: {design_strategy}"
    )
    result_dir = os.path.join(exp_dir, f"{package_geometry}_{design_strategy}")
    os.makedirs(result_dir, exist_ok=True)
    package_size = (100, 100)

    if os.path.exists(os.path.join(result_dir, "package.pkl")):
        package = Package.from_pickle(os.path.join(result_dir, "package.pkl"))
    else:
        if design_strategy == "baseline":
            package = Package(
                package_size,
                design_scheme="bottom_up",
                design_strategy="manual",
                bottom_up_init_type=np.ones(package_size, dtype=np.int8),
                result_dir=result_dir,
                fab_model=FabricateModel(compute_geometry=package_geometry),
            )
        elif design_strategy == "heuristic":
            package = Package(
                package_size,
                design_strategy="heuristic",
                result_dir=result_dir,
                fab_model=FabricateModel(compute_geometry=package_geometry),
            )
        elif design_strategy == "greedy":
            package = Package(
                package_size,
                design_strategy="greedy",
                result_dir=result_dir,
                fab_model=FabricateModel(compute_geometry=package_geometry),
            )
        elif design_strategy == "rl":
            env = QuNoCEnv(package_size, patience=5, result_dir=result_dir)
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
        writer.writerow([package_geometry, design_strategy, score, fidelity, fide_std])
    logging.info(f"Tested score: {score}, fidelity: {fidelity}, fide_std: {fide_std}")
