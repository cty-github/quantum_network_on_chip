import argparse
import copy
import csv
import json
import logging
import math
import os
import sys
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Any

sys.path.append(str(Path(__file__).resolve().parents[1]))

import yaml

from qunoc.element.benchmark import get_program
from qunoc.element.package import FabricateModel, Package
from qunoc.routing.router import Router


@dataclass
class ExperimentSpec:
    name: str
    exp_dir: str
    log_filename: str
    output_csv: str
    defaults: dict[str, Any] = field(default_factory=dict)
    scalability: dict[str, list[Any]] = field(default_factory=dict)


@dataclass
class ExperimentConfig:
    exp_dir: str = "./result/routing_scalability"
    log_filename: str = "log"
    output_csv: str = "output.csv"
    defaults: dict[str, Any] = field(default_factory=dict)
    experiments: dict[str, ExperimentSpec] = field(default_factory=dict)


DEFAULT_CONFIG: dict[str, Any] = {
    "exp_dir": "./result/routing_scalability",
    "log_filename": "log",
    "output_csv": "output.csv",
    "defaults": {
        "layout_root_dir": "./result/layout_scalability",
        "routing_algorithms": ["vqe", "qft", "qaoa", "bv"],
        "layout_design_strategy": "greedy",
        "package_area": 10000,
        "reuse_result": True,
    },
    "experiments": {
        "package_size": {
            "exp_dir": "./result/routing_scalability/package_size",
            "output_csv": "output.csv",
            "defaults": {
                "package_size": [100, 100],
            },
            "scalability": {
                "layout_design_strategy_to_test": ["template", "heuristic", "greedy", "rl"],
                "package_size_to_test": [40, 60, 80, 100, 120],
                "algorithm_to_test": ["vqe", "qft", "qaoa", "bv"],
            },
        },
        "package_shape": {
            "exp_dir": "./result/routing_scalability/package_shape",
            "output_csv": "output.csv",
            "defaults": {
                "package_area": 10000,
                "layout_design_strategy": "heuristic",
            },
            "scalability": {
                "package_x_length_to_test": [10, 20, 40, 50, 80, 100],
                "algorithm_to_test": ["vqe", "qft", "qaoa", "bv"],
            },
        },
        "geometry": {
            "exp_dir": "./result/routing_scalability/geometry",
            "output_csv": "output.csv",
            "defaults": {
                "package_size": [100, 100],
                "layout_design_strategy": "heuristic",
            },
            "scalability": {
                "geometry_to_test": ["square", "hexagon", "heavy_square", "heavy_hexagon"],
                "algorithm_to_test": ["vqe", "qft", "qaoa", "bv"],
            },
        },
        "inter_fidelity": {
            "exp_dir": "./result/routing_scalability/inter_fidelity",
            "output_csv": "output.csv",
            "defaults": {
                "package_size": [100, 100],
                "layout_design_strategy": "heuristic",
            },
            "scalability": {
                "inter_fidelity_to_test": [0.96, 0.965, 0.97, 0.975, 0.98, 0.985, 0.99],
                "algorithm_to_test": ["vqe", "qft", "qaoa", "bv"],
            },
        },
        "chiplet_sparsity": {
            "exp_dir": "./result/routing_scalability/chiplet_sparsity",
            "output_csv": "output.csv",
            "defaults": {
                "package_size": [100, 100],
                "layout_design_strategy": "heuristic",
            },
            "scalability": {
                "chiplet_sparsity_to_test": [1, 3, 5, 7],
                "algorithm_to_test": ["vqe", "qft", "qaoa", "bv"],
            },
        },
    },
}

AXIS_TARGETS = {
    "layout_design_strategy_to_test": "layout_design_strategy",
    "package_size_to_test": "package_size",
    "package_x_length_to_test": "package_x_size",
    "geometry_to_test": "fab_model.compute_geometry",
    "inter_fidelity_to_test": "fab_model.inter_fide",
    "chiplet_sparsity_to_test": "fab_model.inter_sparsity",
    "algorithm_to_test": "algorithm",
}

AXIS_LABELS = {
    "layout_design_strategy_to_test": "LayoutDesignStrategy",
    "package_size_to_test": "PackageSize",
    "package_x_length_to_test": "PackageXLength",
    "geometry_to_test": "PackageGeometry",
    "inter_fidelity_to_test": "InterFidelity",
    "chiplet_sparsity_to_test": "ChipletSparsity",
    "algorithm_to_test": "Algorithm",
}


def _deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_update(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _normalize_package_size(value: Any) -> tuple[int, int]:
    if isinstance(value, int):
        return value, value
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return int(value[0]), int(value[1])
    raise ValueError(f"Invalid package size value: {value!r}")


def _format_value(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return "x".join(str(item) for item in value)
    return str(value)


def _resolve_output_path(path_value: str | os.PathLike[str], *, base_dir: Path) -> str:
    path = Path(path_value)
    if not path.is_absolute():
        path = base_dir / path
    return str(path.resolve())



def _set_nested(config: dict[str, Any], key_path: str, value: Any) -> None:
    parts = key_path.split(".")
    current = config
    for part in parts[:-1]:
        current = current.setdefault(part, {})
    current[parts[-1]] = value


def load_experiment_config(path: str | Path | None) -> ExperimentConfig:
    default_raw = copy.deepcopy(DEFAULT_CONFIG)
    raw = default_raw
    config_dir = Path(__file__).resolve().parent
    if path is not None:
        config_path = Path(path).resolve()
        config_dir = config_path.parent
        with config_path.open("r", encoding="utf-8") as file:
            if config_path.suffix.lower() in {".yaml", ".yml"}:
                user_raw = yaml.safe_load(file) or {}
            else:
                user_raw = json.load(file) or {}
        if not isinstance(user_raw, dict):
            raise ValueError("Config file must contain a mapping at the top level.")
        # If user provided an explicit `experiments` section, replace defaults' experiments
        user_exps = copy.deepcopy(user_raw.get("experiments")) if isinstance(user_raw, dict) else None
        # Merge other keys but avoid merging default experiments when user specified experiments
        user_no_exps = {k: v for k, v in user_raw.items() if k != "experiments"}
        raw = _deep_update(default_raw, user_no_exps)
        if user_exps is not None:
            raw["experiments"] = copy.deepcopy(user_exps)

    root_exp_dir = _resolve_output_path(
        raw.get("exp_dir", DEFAULT_CONFIG["exp_dir"]),
        base_dir=config_dir,
    )
    defaults = copy.deepcopy(raw.get("defaults", {}))
    if "layout_root_dir" in defaults:
        defaults["layout_root_dir"] = _resolve_output_path(
            defaults["layout_root_dir"],
            base_dir=config_dir,
        )

    experiments: dict[str, ExperimentSpec] = {}
    for experiment_name, experiment_raw in raw.get("experiments", {}).items():
        if not isinstance(experiment_raw, dict):
            continue
        experiment_exp_dir = experiment_raw.get(
            "exp_dir",
            os.path.join(root_exp_dir, experiment_name),
        )
        experiments[experiment_name] = ExperimentSpec(
            name=experiment_name,
            exp_dir=_resolve_output_path(experiment_exp_dir, base_dir=config_dir),
            log_filename=str(experiment_raw.get("log_filename", raw.get("log_filename", DEFAULT_CONFIG["log_filename"]))),
            output_csv=str(experiment_raw.get("output_csv", raw.get("output_csv", DEFAULT_CONFIG["output_csv"]))),
            defaults=copy.deepcopy(experiment_raw.get("defaults", {})),
            scalability=copy.deepcopy(experiment_raw.get("scalability", {})),
        )

    return ExperimentConfig(
        exp_dir=root_exp_dir,
        log_filename=str(raw.get("log_filename", DEFAULT_CONFIG["log_filename"])),
        output_csv=str(raw.get("output_csv", DEFAULT_CONFIG["output_csv"])),
        defaults=defaults,
        experiments=experiments,
    )


def _build_sweep_axes(scalability: dict[str, list[Any]]) -> tuple[list[str], list[list[tuple[str, Any]]]]:
    axis_names: list[str] = []
    axis_values: list[list[tuple[str, Any]]] = []

    for raw_key, values in scalability.items():
        if not raw_key.endswith("_to_test"):
            continue

        target_key = AXIS_TARGETS.get(raw_key, raw_key[: -len("_to_test")])
        normalized_values: list[Any] = []
        for value in values:
            if target_key == "package_size":
                normalized_values.append(int(value))
            elif target_key == "package_x_size":
                normalized_values.append(int(value))
            elif target_key == "layout_design_strategy":
                normalized_values.append(str(value))
            elif target_key == "fab_model.compute_geometry":
                normalized_values.append(str(value))
            elif target_key == "fab_model.inter_fide":
                normalized_values.append(float(value))
            elif target_key == "fab_model.inter_sparsity":
                normalized_values.append(int(value))
            elif target_key == "algorithm":
                normalized_values.append(str(value))
            else:
                normalized_values.append(value)

        axis_names.append(raw_key)
        axis_values.append([(target_key, value) for value in normalized_values])

    return axis_names, axis_values


def _resolve_package_size(run_config: dict[str, Any]) -> tuple[int, int]:
    if "package_size" in run_config:
        return _normalize_package_size(run_config["package_size"])
    if "package_x_size" in run_config:
        package_x_size = int(run_config["package_x_size"])
        package_area = int(run_config.get("package_area", 10000))
        if package_x_size <= 0:
            raise ValueError("package_x_size must be positive.")
        if package_area % package_x_size != 0:
            raise ValueError(
                f"package_area {package_area} is not divisible by package_x_size {package_x_size}."
            )
        return package_x_size, package_area // package_x_size
    raise KeyError("run_config must define package_size or package_x_size.")


def _data_qubit_num(package_size: tuple[int, int]) -> int:
    node_num = math.prod(package_size)
    return int(100 * 2 ** math.log(node_num / 1600, 5))


def _build_layout_package_path(common_defaults: dict[str, Any], spec_name: str, run_config: dict[str, Any]) -> str:
    layout_root_dir = common_defaults.get("layout_root_dir", "./result/layout_scalability")
    layout_dir = os.path.join(layout_root_dir, spec_name)
    layout_strategy = str(run_config.get("layout_design_strategy", common_defaults.get("layout_design_strategy", "greedy")))
    package_size = _resolve_package_size(run_config)
    if spec_name == "package_size":
        return os.path.join(layout_dir, f"{package_size[0]}_{layout_strategy}", "package.pkl")
    if spec_name == "package_shape":
        return os.path.join(layout_dir, f"{package_size[0]}_{package_size[1]}_{layout_strategy}", "package.pkl")
    if spec_name == "geometry":
        geometry = str(run_config.get("fab_model", {}).get("compute_geometry", "square"))
        return os.path.join(layout_dir, f"{geometry}_{layout_strategy}", "package.pkl")
    if spec_name == "inter_fidelity":
        inter_fide = run_config.get("fab_model", {}).get("inter_fide")
        return os.path.join(layout_dir, f"{inter_fide}_{layout_strategy}", "package.pkl")
    if spec_name == "chiplet_sparsity":
        inter_sparsity = run_config.get("fab_model", {}).get("inter_sparsity")
        return os.path.join(layout_dir, f"{inter_sparsity}_{layout_strategy}", "package.pkl")

    raise ValueError(f"Unsupported routing experiment: {spec_name}")


def _build_result_dir_name(spec_name: str, run_config: dict[str, Any], algorithm: str) -> str:
    layout_strategy = str(run_config.get("layout_design_strategy", "greedy"))
    package_size = _resolve_package_size(run_config)
    if spec_name == "package_size":
        return f"{package_size[0]}_{layout_strategy}_{algorithm}"
    if spec_name == "package_shape":
        return f"{package_size[0]}_{package_size[1]}_{layout_strategy}_{algorithm}"
    if spec_name == "geometry":
        geometry = str(run_config.get("fab_model", {}).get("compute_geometry", "square"))
        return f"{geometry}_{layout_strategy}_{algorithm}"
    if spec_name == "inter_fidelity":
        inter_fide = run_config.get("fab_model", {}).get("inter_fide")
        return f"{inter_fide}_{layout_strategy}_{algorithm}"
    if spec_name == "chiplet_sparsity":
        inter_sparsity = run_config.get("fab_model", {}).get("inter_sparsity")
        return f"{inter_sparsity}_{layout_strategy}_{algorithm}"
    return f"{layout_strategy}_{algorithm}"


def _route_once(common_defaults: dict[str, Any], spec: ExperimentSpec, run_config: dict[str, Any]) -> dict[str, Any]:
    package_path = _build_layout_package_path(common_defaults, spec.name, run_config)
    if not os.path.exists(package_path):
        raise FileNotFoundError(
            f"Missing exact layout package: {package_path}. "
            "Run evaluate/test_layout_scalability.py with matching layout parameters first."
        )

    package = Package.from_pickle(filepath=package_path)
    package_size = package.package_size
    data_qubit_num = _data_qubit_num(package_size)
    algorithm = str(run_config["algorithm"])
    program = get_program(algorithm, data_qubit_num)

    result_dir = os.path.join(spec.exp_dir, _build_result_dir_name(spec.name, run_config, algorithm))
    os.makedirs(result_dir, exist_ok=True)

    cache_path = os.path.join(result_dir, "exp_result.json")
    circuit_path = os.path.join(result_dir, "qunoc_circuit.pkl")
    if bool(run_config.get("reuse_result", True)) and os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as file:
            return json.load(file)

    router = Router(package, program, result_dir=result_dir)
    circuit = router.route_program_on_package()
    circuit.to_pickle(circuit_path)
    intra_gate_num, inter_gate_num, qst_dist, meas_num, total_fide = router.statistic_route_results()
    circuit_duration = router.estimate_qunoc_circuit_duration()
    logic_gate_fide = router.estimate_qunoc_logic_gate_fide()

    exp_result = {
        "algorithm": algorithm,
        "data_qubit_num": data_qubit_num,
        "logic_gate_num": program.get_gate_depend_graph().number_of_nodes() - 1,
        "circuit_duration": circuit_duration,
        "gate_fidelity": logic_gate_fide,
        "intra_gate": intra_gate_num,
        "inter_gate": inter_gate_num,
        "qst_dist": qst_dist,
        "meas_num": meas_num,
        "total_fide": total_fide,
        "layout_package_path": package_path,
    }

    with open(cache_path, "w", encoding="utf-8") as file:
        json.dump(exp_result, file, ensure_ascii=False, indent=2)

    return exp_result


def _run_experiment(common_defaults: dict[str, Any], spec: ExperimentSpec) -> None:
    os.makedirs(spec.exp_dir, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(spec.exp_dir, f"{spec.log_filename}.txt"),
        filemode="w",
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
        encoding="utf-8",
        force=True,
    )

    logging.info("Experiment: %s", spec.name)
    logging.info("Defaults: %s", spec.defaults)
    logging.info("Scalability: %s", spec.scalability)

    run_defaults = _deep_update(common_defaults, spec.defaults)
    axis_names, axis_values = _build_sweep_axes(spec.scalability)
    if not axis_names:
        logging.info("Skip %s because no sweep axes were provided.", spec.name)
        return

    output_path = os.path.join(spec.exp_dir, spec.output_csv)
    if not os.path.exists(output_path):
        with open(output_path, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(
                [AXIS_LABELS.get(name, name) for name in axis_names]
                + [
                    "Algorithm",
                    "#DataQubit",
                    "#LogicGate",
                    "CircuitDuration",
                    "GateFidelity",
                    "IntraGate",
                    "InterGate",
                    "QSTDist",
                    "MeasNum",
                    "TotalFide",
                ]
            )

    for values in product(*axis_values):
        run_config = copy.deepcopy(run_defaults)
        for target_key, value in values:
            _set_nested(run_config, target_key, value)

        exp_result = _route_once(common_defaults, spec, run_config)

        with open(output_path, mode="a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    *(_format_value(value) for _, value in values),
                    exp_result["algorithm"],
                    exp_result["data_qubit_num"],
                    exp_result["logic_gate_num"],
                    exp_result["circuit_duration"],
                    exp_result["gate_fidelity"],
                    exp_result["intra_gate"],
                    exp_result["inter_gate"],
                    exp_result["qst_dist"],
                    exp_result["meas_num"],
                    exp_result["total_fide"],
                ]
            )

        logging.info(
            "Finished %s -> duration=%s, fidelity=%s",
            _build_result_dir_name(spec.name, run_config, exp_result["algorithm"]),
            exp_result["circuit_duration"],
            exp_result["gate_fidelity"],
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified routing scalability test runner.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).with_name("test_routing_scalability.yaml")),
        help="Path to a JSON/YAML config file.",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_file():
        fallback = Path(__file__).with_name(config_path.name)
        if fallback.is_file():
            config_path = fallback

    cfg = load_experiment_config(config_path if config_path.is_file() else None)
    if not cfg.experiments:
        raise ValueError("No experiments were defined in the config file.")

    for spec in cfg.experiments.values():
        _run_experiment(cfg.defaults, spec)


if __name__ == "__main__":
    main()