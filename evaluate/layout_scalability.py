import argparse
import copy
import csv
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Any

sys.path.append("..")

import yaml

from qunoc.element.package import FabricateModel, Package, np
from qunoc.layout.agent import LayoutAgent
from qunoc.layout.env import QuNoCEnv


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
    exp_dir: str = "./result/layout_scalability"
    log_filename: str = "log"
    output_csv: str = "output.csv"
    defaults: dict[str, Any] = field(default_factory=dict)
    experiments: dict[str, ExperimentSpec] = field(default_factory=dict)


DEFAULT_CONFIG: dict[str, Any] = {
    "exp_dir": "./result/layout_scalability",
    "log_filename": "log",
    "output_csv": "output.csv",
    "defaults": {
        "package_size": [100, 100],
        "chiplet_length": 10,
        "design_strategy": "greedy",
        "compute_ratio": 0.8,
        "rl_patience": 5,
        "rl_train_timesteps": None,
        "reuse_package": True,
        "save_package": True,
        "save_figure": True,
        "reuse_agent": True,
        "fab_model": {
            "compute_geometry": "square",
            "inter_sparsity": 3,
            "inter_fide": 0.983,
        },
    },
    "experiments": {
        "package_size": {
            "exp_dir": "./result/layout_scalability/package_size",
            "output_csv": "output.csv",
            "defaults": {},
            "scalability": {
                "design_strategy_to_test": ["template", "heuristic", "greedy", "rl"],
                "package_size_to_test": [40, 60, 80, 100, 120],
            },
        },
        "package_shape": {
            "exp_dir": "./result/layout_scalability/package_shape",
            "output_csv": "output.csv",
            "defaults": {
                "package_area": 10000,
            },
            "scalability": {
                "design_strategy_to_test": ["heuristic"],
                "package_x_length_to_test": [10, 20, 40, 50, 80, 100],
            },
        },
        "geometry": {
            "exp_dir": "./result/layout_scalability/geometry",
            "output_csv": "output.csv",
            "defaults": {
                "package_size": [100, 100],
                "fab_model": {
                    "compute_geometry": "square",
                },
            },
            "scalability": {
                "design_strategy_to_test": ["heuristic"],
                "geometry_to_test": ["square", "hexagon", "heavy_square", "heavy_hexagon"],
            },
        },
        "inter_fidelity": {
            "exp_dir": "./result/layout_scalability/inter_fidelity",
            "output_csv": "output.csv",
            "defaults": {
                "package_size": [100, 100],
                "fab_model": {
                    "compute_geometry": "square",
                    "inter_sparsity": 3,
                },
            },
            "scalability": {
                "design_strategy_to_test": ["heuristic"],
                "inter_fidelity_to_test": [0.96, 0.965, 0.97, 0.975, 0.98, 0.985, 0.99],
            },
        },
        "chiplet_sparsity": {
            "exp_dir": "./result/layout_scalability/chiplet_sparsity",
            "output_csv": "output.csv",
            "defaults": {
                "package_size": [100, 100],
                "fab_model": {
                    "compute_geometry": "square",
                    "inter_fide": 0.983,
                },
            },
            "scalability": {
                "design_strategy_to_test": ["heuristic"],
                "chiplet_sparsity_to_test": [1, 3, 5, 7],
            },
        },
    },
}

AXIS_TARGETS = {
    "design_strategy_to_test": "design_strategy",
    "chiplet_length_to_test": "chiplet_length",
    "package_size_to_test": "package_size",
    "package_x_length_to_test": "package_x_size",
    "geometry_to_test": "fab_model.compute_geometry",
    "chiplet_sparsity_to_test": "fab_model.inter_sparsity",
    "inter_fidelity_to_test": "fab_model.inter_fide",
    "inter_fide_to_test": "fab_model.inter_fide",
}

AXIS_LABELS = {
    "design_strategy_to_test": "DesignStrategy",
    "chiplet_length_to_test": "ChipletLength",
    "package_size_to_test": "PackageSize",
    "package_x_length_to_test": "PackageXLength",
    "geometry_to_test": "PackageGeometry",
    "chiplet_sparsity_to_test": "ChipletSparsity",
    "inter_fidelity_to_test": "InterFidelity",
    "inter_fide_to_test": "InterFidelity",
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


def _set_nested(config: dict[str, Any], key_path: str, value: Any) -> None:
    parts = key_path.split(".")
    current = config
    for part in parts[:-1]:
        current = current.setdefault(part, {})
    current[parts[-1]] = value


def load_experiment_config(path: str | Path | None) -> ExperimentConfig:
    raw = copy.deepcopy(DEFAULT_CONFIG)
    if path is not None:
        config_path = Path(path)
        with config_path.open("r", encoding="utf-8") as file:
            if config_path.suffix.lower() in {".yaml", ".yml"}:
                user_raw = yaml.safe_load(file) or {}
            else:
                user_raw = json.load(file) or {}
        if not isinstance(user_raw, dict):
            raise ValueError("Config file must contain a mapping at the top level.")
        raw = _deep_update(raw, user_raw)

    if "defaults" not in raw:
        raw["defaults"] = {}

    if "experiments" not in raw:
        legacy_scalability = raw.pop("scalability", {})
        raw["experiments"] = {
            "package_size": {
                "exp_dir": os.path.join(raw.get("exp_dir", DEFAULT_CONFIG["exp_dir"]), "package_size"),
                "output_csv": raw.get("output_csv", DEFAULT_CONFIG["output_csv"]),
                "defaults": {},
                "scalability": legacy_scalability,
            }
        }

    legacy_fab_keys = {
        "compute_geometry",
        "inter_sparsity",
        "inter_fide",
        "max_capacity",
        "max_length",
        "single_fide",
        "enhanced_fide",
        "yield_5",
        "yield_decay",
    }
    legacy_fab_model = raw["defaults"].setdefault("fab_model", {})
    for key in legacy_fab_keys:
        if key in raw and key not in legacy_fab_model:
            legacy_fab_model[key] = raw[key]
    if "init_mapping" in raw and "init_mapping" not in raw["defaults"]:
        raw["defaults"]["init_mapping"] = raw["init_mapping"]

    experiments: dict[str, ExperimentSpec] = {}
    for experiment_name, experiment_raw in raw.get("experiments", {}).items():
        if not isinstance(experiment_raw, dict):
            continue
        experiments[experiment_name] = ExperimentSpec(
            name=experiment_name,
            exp_dir=str(experiment_raw.get("exp_dir", os.path.join(raw.get("exp_dir", DEFAULT_CONFIG["exp_dir"]), experiment_name))),
            log_filename=str(experiment_raw.get("log_filename", raw.get("log_filename", DEFAULT_CONFIG["log_filename"]))),
            output_csv=str(experiment_raw.get("output_csv", raw.get("output_csv", DEFAULT_CONFIG["output_csv"]))),
            defaults=copy.deepcopy(experiment_raw.get("defaults", {})),
            scalability=copy.deepcopy(experiment_raw.get("scalability", {})),
        )

    return ExperimentConfig(
        exp_dir=str(raw.get("exp_dir", DEFAULT_CONFIG["exp_dir"])),
        log_filename=str(raw.get("log_filename", DEFAULT_CONFIG["log_filename"])),
        output_csv=str(raw.get("output_csv", DEFAULT_CONFIG["output_csv"])),
        defaults=copy.deepcopy(raw.get("defaults", {})),
        experiments=experiments,
    )


def _build_sweep_axes(
    scalability: dict[str, list[Any]],
) -> tuple[list[str], list[list[tuple[str, Any]]]]:
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
            elif target_key == "chiplet_length":
                normalized_values.append(int(value))
            elif target_key == "design_strategy":
                normalized_values.append(str(value))
            elif target_key == "fab_model.compute_geometry":
                normalized_values.append(str(value))
            elif target_key == "fab_model.inter_sparsity":
                normalized_values.append(int(value))
            elif target_key == "fab_model.inter_fide":
                normalized_values.append(float(value))
            else:
                normalized_values.append(value)

        axis_names.append(raw_key)
        axis_values.append([(target_key, value) for value in normalized_values])

    return axis_names, axis_values


def _build_result_dir_name(experiment_name: str, run_config: dict[str, Any]) -> str:
    package_size = _resolve_package_size(run_config)
    design_strategy = str(run_config.get("design_strategy", "greedy"))

    if experiment_name == "package_size":
        return f"{package_size[0]}_{design_strategy}"
    if experiment_name == "package_shape":
        return f"{package_size[0]}_{package_size[1]}_{design_strategy}"
    if experiment_name == "geometry":
        geometry = str(run_config.get("fab_model", {}).get("compute_geometry", "square"))
        return f"{geometry}_{design_strategy}"
    if experiment_name == "inter_fidelity":
        inter_fide = run_config.get("fab_model", {}).get("inter_fide", "unknown")
        return f"{inter_fide}_{design_strategy}"
    if experiment_name == "chiplet_sparsity":
        sparsity = run_config.get("fab_model", {}).get("inter_sparsity", "unknown")
        return f"{sparsity}_{design_strategy}"

    return "_".join(
        [
            _format_value(package_size),
            design_strategy,
            *(_format_value(value) for _, value in run_config.items()),
        ]
    )


def _build_fab_model(run_config: dict[str, Any]) -> FabricateModel:
    fab_model_kwargs = copy.deepcopy(run_config.get("fab_model", {}))
    chiplet_length = int(run_config.get("chiplet_length", 10))
    fab_model_kwargs.setdefault("max_capacity", chiplet_length**2)
    fab_model_kwargs.setdefault("max_length", int(0.8 * chiplet_length**2))
    return FabricateModel(**fab_model_kwargs)


def _resolve_package_size(run_config: dict[str, Any]) -> tuple[int, int]:
    if "package_size" in run_config:
        return _normalize_package_size(run_config["package_size"])
    if "package_x_size" in run_config:
        package_x_size = int(run_config["package_x_size"])
        package_area = int(run_config.get("package_area", 10000))
        if package_x_size <= 0:
            raise ValueError("package_x_size must be positive.")
        return package_x_size, package_area // package_x_size
    raise KeyError("run_config must define package_size or package_x_size.")


def _build_package(
    run_config: dict[str, Any],
    fab_model: FabricateModel,
    result_dir: str,
) -> Package:
    package_size = _resolve_package_size(run_config)
    design_strategy = str(run_config["design_strategy"])
    compute_ratio = float(run_config.get("compute_ratio", 0.8))

    if design_strategy == "baseline":
        return Package(
            package_size=package_size,
            fab_model=fab_model,
            design_scheme="bottom_up",
            design_strategy="manual",
            bottom_up_init_type=np.ones(package_size, dtype=np.int8),
            result_dir=result_dir,
        )
    if design_strategy == "template":
        return Package(
            package_size=package_size,
            fab_model=fab_model,
            design_strategy="template",
            compute_ratio=compute_ratio,
            result_dir=result_dir,
        )
    if design_strategy in {"stripe", "grid", "radial", "heuristic", "greedy"}:
        return Package(
            package_size=package_size,
            fab_model=fab_model,
            design_strategy=design_strategy,
            compute_ratio=compute_ratio,
            result_dir=result_dir,
        )
    if design_strategy == "rl":
        env = QuNoCEnv(
            package_size=package_size,
            fab_model=fab_model,
            patience=int(run_config.get("rl_patience", 5)),
            result_dir=result_dir,
        )
        model_path = os.path.join(result_dir, "layout_agent.pth")
        if run_config.get("reuse_agent", True) and os.path.exists(model_path):
            agent = LayoutAgent(env, pre_trained_model_path=model_path, result_dir=result_dir)
        else:
            agent = LayoutAgent(env, result_dir=result_dir)
            total_timesteps = run_config.get("rl_train_timesteps")
            if total_timesteps is None:
                total_timesteps = 100 * package_size[0] * package_size[1]
            agent.train(total_timesteps=int(total_timesteps))
        return agent.design()

    raise ValueError(f"Unsupported design strategy: {design_strategy}")


def _maybe_load_cached_package(result_dir: str, reuse_package: bool) -> Package | None:
    package_path = os.path.join(result_dir, "package.pkl")
    if reuse_package and os.path.exists(package_path):
        return Package.from_pickle(package_path)
    return None


def _persist_package(package: Package, result_dir: str, save_package: bool, save_figure: bool) -> None:
    if save_package:
        package.to_pickle()
        package.save_design_history()
    if save_figure:
        try:
            figure, _ = package.draw_qubits(axis_off=False)
            figure.savefig(os.path.join(result_dir, "package.png"))
            logging.info("Saved package visualization")
        except Exception as error:
            logging.warning(f"Failed to draw qubits: {error}")


def _record_design_failure(result_dir: str, error: Exception) -> None:
    failure_path = os.path.join(result_dir, "design_failed.json")
    try:
        with open(failure_path, mode="w", encoding="utf-8") as file:
            json.dump({"error": str(error), "type": type(error).__name__}, file, ensure_ascii=False, indent=2)
    except Exception as write_error:
        logging.warning("Failed to write design failure file for %s: %s", result_dir, write_error)


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
            writer.writerow([AXIS_LABELS.get(name, name) for name in axis_names] + ["Score", "Fidelity", "FideStd"])

    for values in product(*axis_values):
        run_config = copy.deepcopy(run_defaults)
        for target_key, value in values:
            _set_nested(run_config, target_key, value)

        result_dir_name = _build_result_dir_name(spec.name, run_config)
        result_dir = os.path.join(spec.exp_dir, result_dir_name)
        os.makedirs(result_dir, exist_ok=True)

        fab_model = _build_fab_model(run_config)
        try:
            package = _maybe_load_cached_package(
                result_dir,
                bool(run_config.get("reuse_package", True)),
            )
            if package is None:
                package = _build_package(run_config, fab_model, result_dir)
                _persist_package(
                    package,
                    result_dir,
                    bool(run_config.get("save_package", True)),
                    bool(run_config.get("save_figure", True)),
                )
        except Exception as error:
            logging.exception("Failed to build package for %s", result_dir_name)
            _record_design_failure(result_dir, error)
            continue

        score = package.estimate_score()
        fidelity, fide_std = package.estimate_transfer()
        with open(output_path, mode="a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow([
                *(_format_value(value) for _, value in values),
                score,
                fidelity,
                fide_std,
            ])

        logging.info(
            "Finished %s -> score=%s, fidelity=%s, fide_std=%s",
            result_dir_name,
            score,
            fidelity,
            fide_std,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified layout generation test runner.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).with_name("layout_scalability.yaml")),
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
