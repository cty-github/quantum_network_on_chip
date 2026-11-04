import argparse
import copy
import csv
import json
import logging
import os
import random
import sys
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Any

sys.path.append(str(Path(__file__).resolve().parents[1]))

import yaml
from qunoc.element.package import FabricateModel, Package, np
from qunoc.layout.agent import LayoutAgent
from qunoc.layout.env import QuNoCEnv
from qunoc.layout.schema import read_model_manifest


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
    config_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent)


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
        "rl_model_path": None,
        "rl_estimate_workers": 1,
        "rl_seed": None,
        "rl_inference_seed": 1042,
        "rl_score_seeds": [0],
        "rl_canvas_size": None,
        "rl_max_module_slots": None,
        "rl_relative_interval_bins": None,
        "rl_allow_legacy_model": False,
        "reuse_package": True,
        "save_package": True,
        "save_figure": True,
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


def _resolve_path(path_value: str | None, *, base_dir: Path) -> Path | None:
    if path_value is None:
        return None
    path = Path(path_value)
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


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
    if path is not None:
        config_path = Path(path)
        with config_path.open("r", encoding="utf-8") as file:
            if config_path.suffix.lower() in {".yaml", ".yml"}:
                user_raw = yaml.safe_load(file) or {}
            else:
                user_raw = json.load(file) or {}
        if not isinstance(user_raw, dict):
            raise ValueError("Config file must contain a mapping at the top level.")
        user_experiments = copy.deepcopy(user_raw.get("experiments"))
        user_without_experiments = {
            key: value for key, value in user_raw.items() if key != "experiments"
        }
        raw = _deep_update(default_raw, user_without_experiments)
        if user_experiments is not None:
            raw["experiments"] = user_experiments

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

    config_dir = Path(path).resolve().parent if path is not None else Path(__file__).resolve().parent
    root_exp_dir = _resolve_output_path(
        raw.get("exp_dir", DEFAULT_CONFIG["exp_dir"]),
        base_dir=config_dir,
    )

    experiments: dict[str, ExperimentSpec] = {}
    default_experiments = DEFAULT_CONFIG.get("experiments", {})
    for experiment_name, experiment_raw in raw.get("experiments", {}).items():
        if not isinstance(experiment_raw, dict):
            continue
        experiment_override = copy.deepcopy(experiment_raw)
        for optional_section in ("defaults", "scalability"):
            if experiment_override.get(optional_section) is None:
                experiment_override.pop(optional_section, None)
        experiment_config = _deep_update(
            copy.deepcopy(default_experiments.get(experiment_name, {})),
            experiment_override,
        )
        experiment_exp_dir = experiment_config.get(
            "exp_dir",
            os.path.join(root_exp_dir, experiment_name),
        )
        experiments[experiment_name] = ExperimentSpec(
            name=experiment_name,
            exp_dir=_resolve_output_path(experiment_exp_dir, base_dir=config_dir),
            log_filename=str(experiment_config.get("log_filename", raw.get("log_filename", DEFAULT_CONFIG["log_filename"]))),
            output_csv=str(experiment_config.get("output_csv", raw.get("output_csv", DEFAULT_CONFIG["output_csv"]))),
            defaults=copy.deepcopy(experiment_config.get("defaults", {})),
            scalability=copy.deepcopy(experiment_config.get("scalability", {})),
        )

    defaults = copy.deepcopy(raw.get("defaults", {}))
    defaults["_config_dir"] = str(config_dir)
    return ExperimentConfig(
        exp_dir=root_exp_dir,
        log_filename=str(raw.get("log_filename", DEFAULT_CONFIG["log_filename"])),
        output_csv=str(raw.get("output_csv", DEFAULT_CONFIG["output_csv"])),
        defaults=defaults,
        experiments=experiments,
        config_dir=config_dir,
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
    if "package_x_size" in run_config:
        package_x_size = int(run_config["package_x_size"])
        package_area = int(run_config.get("package_area", 10000))
        if package_x_size <= 0:
            raise ValueError("package_x_size must be positive.")
        return package_x_size, package_area // package_x_size
    if "package_size" in run_config:
        return _normalize_package_size(run_config["package_size"])
    raise KeyError("run_config must define package_size or package_x_size.")


def _build_rl_env_kwargs(
    run_config: dict[str, Any],
    fab_model: FabricateModel,
    package_size: tuple[int, int],
    result_dir: str | None,
) -> dict[str, Any]:
    seed = run_config.get("rl_seed")
    seed = int(seed) if seed is not None else 0
    score_seed = run_config.get("rl_score_seed")
    score_seed = int(score_seed) if score_seed is not None else seed
    score_seeds = run_config.get("rl_score_seeds") or [score_seed]
    canvas_value = run_config.get("rl_canvas_size") or package_size
    slot_value = run_config.get("rl_max_module_slots") or 256
    interval_value = run_config.get("rl_relative_interval_bins") or [0.08, 0.12, 0.16, 0.22, 0.30, 0.40]
    return {
        "package_size": package_size,
        "fab_model": fab_model,
        "patience": int(run_config.get("rl_patience", 5)),
        "result_dir": result_dir,
        "estimate_workers": int(run_config.get("rl_estimate_workers", 1)),
        "compute_ratio": float(run_config.get("compute_ratio", 0.8)),
        "score_seed": score_seed,
        "score_seeds": [int(value) for value in score_seeds],
        "invalid_action_penalty": float(
            run_config.get("rl_invalid_action_penalty", -2.0)
        ),
        "canvas_size": _normalize_package_size(canvas_value),
        "max_module_slots": int(slot_value),
        "relative_interval_bins": tuple(float(value) for value in interval_value),
        "base_seed": seed,
        "record_design_history": bool(run_config.get("rl_record_design_history", False)),
    }


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
        model_path = _resolve_path(
            run_config.get("rl_model_path"),
            base_dir=Path(run_config.get("_config_dir", Path(__file__).resolve().parent)),
        )
        if model_path is None:
            raise ValueError(
                "RL layout inference requires rl_model_path. Train first with "
                "evaluate/train_layout_model.py; test_layout_scalability.py no longer trains models."
            )
        if not model_path.exists():
            raise FileNotFoundError(f"Missing RL layout model: {model_path}")
        model_manifest = read_model_manifest(model_path)
        if model_manifest is None:
            raise ValueError(
                f"RL layout model {model_path} has no model_manifest.json. "
                "Inference-only scalability does not load legacy checkpoints."
            )
        schema = model_manifest.get("schema", {})
        run_config = copy.deepcopy(run_config)
        run_config["rl_canvas_size"] = schema.get(
            "canvas_size", run_config.get("rl_canvas_size", package_size)
        )
        run_config["rl_max_module_slots"] = schema.get(
            "max_module_slots", run_config.get("rl_max_module_slots", 256)
        )
        run_config["rl_relative_interval_bins"] = schema.get(
            "relative_interval_bins",
            run_config.get("rl_relative_interval_bins", [0.08, 0.12, 0.16, 0.22, 0.30, 0.40]),
        )
        run_config["rl_record_design_history"] = False
        design_env = QuNoCEnv(
            **_build_rl_env_kwargs(
                run_config,
                fab_model,
                package_size,
                result_dir,
            )
        )
        try:
            agent = LayoutAgent(
                design_env,
                pre_trained_model_path=str(model_path),
                design_env=design_env,
                allow_legacy_model=bool(run_config.get("rl_allow_legacy_model", False)),
            )
            return agent.design(
                env=design_env,
                seed=int(run_config.get("rl_inference_seed", run_config.get("rl_seed") or 1042)),
            )
        finally:
            design_env.close()

    raise ValueError(f"Unsupported design strategy: {design_strategy}")


def _maybe_load_cached_package(result_dir: str, reuse_package: bool) -> Package | None:
    package_path = os.path.join(result_dir, "package.pkl")
    if not reuse_package or not os.path.exists(package_path):
        return None
    return Package.from_pickle(package_path)


def _persist_package(package: Package, result_dir: str, save_package: bool, save_figure: bool) -> None:
    if save_package:
        package.to_pickle()
        if getattr(package, "record_design_history", True):
            package.save_design_history(save_gif=False)
    if save_figure:
        try:
            figure, _ = package.draw_qubits(axis_off=False)
            figure.savefig(os.path.join(result_dir, "package.png"))
            logging.info("Saved package visualization")
        except Exception as error:
            logging.warning(f"Failed to draw qubits: {error}")


def _evaluate_package(
    package: Package, score_seed: int | None = None
) -> tuple[float, float, float]:
    if score_seed is None:
        score = package.estimate_score()
        fidelity, fide_std = package.estimate_transfer()
        return score, fidelity, fide_std

    python_random_state = random.getstate()
    numpy_random_state = np.random.get_state()
    random.seed(score_seed)
    np.random.seed(score_seed)
    try:
        score = package.estimate_score()
        fidelity, fide_std = package.estimate_transfer()
        return score, fidelity, fide_std
    finally:
        random.setstate(python_random_state)
        np.random.set_state(numpy_random_state)


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
            if target_key == "package_x_size":
                run_config.pop("package_size", None)
            _set_nested(run_config, target_key, value)

        result_dir_name = _build_result_dir_name(spec.name, run_config)
        result_dir = os.path.join(spec.exp_dir, result_dir_name)
        os.makedirs(result_dir, exist_ok=True)

        fab_model = _build_fab_model(run_config)
        try:
            package = None
            if run_config.get("design_strategy") != "rl":
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

        score_seed = None
        if run_config.get("design_strategy") == "rl":
            rl_score_seed = run_config.get("rl_score_seed")
            if rl_score_seed is None:
                score_seeds = run_config.get("rl_score_seeds") or []
                rl_score_seed = score_seeds[0] if score_seeds else None
            score_seed = int(rl_score_seed) if rl_score_seed is not None else None
        score, fidelity, fide_std = _evaluate_package(package, score_seed)
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


def _run_shared_rl(config_path: Path) -> None:
    raise RuntimeError(
        "--mode shared-rl has been removed from test_layout_scalability.py. "
        "Use `python evaluate/train_layout_model.py --config evaluate/train_layout_model.yaml` "
        "to train a model, then run test_layout_scalability.py with rl_model_path for design."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified layout generation test runner.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).with_name("test_layout_scalability.yaml")),
        help="Path to a JSON/YAML config file.",
    )
    parser.add_argument(
        "--mode",
        choices=("sweep", "shared-rl"),
        default="sweep",
        help="Run sweeps. The legacy shared-rl mode now raises with migration instructions.",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_file():
        fallback = Path(__file__).with_name(config_path.name)
        if fallback.is_file():
            config_path = fallback

    if args.mode == "shared-rl":
        if not config_path.is_file():
            raise FileNotFoundError(f"Missing shared RL config: {config_path}")
        _run_shared_rl(config_path)
        return

    cfg = load_experiment_config(config_path if config_path.is_file() else None)
    if not cfg.experiments:
        raise ValueError("No experiments were defined in the config file.")

    for spec in cfg.experiments.values():
        _run_experiment(cfg.defaults, spec)


if __name__ == "__main__":
    main()
