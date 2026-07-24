import argparse
import copy
import logging
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.append(str(Path(__file__).resolve().parents[1]))

import yaml
from stable_baselines3.common.vec_env import SubprocVecEnv

from qunoc.layout.agent import LayoutAgent
from qunoc.layout.artifacts import shared_layout_model_paths, write_json_atomic
from qunoc.layout.env import QuNoCEnv
from qunoc.layout.training import (
    LayoutEnvFactory,
    build_layout_tasks,
    prepare_subprocess_thread_limits,
    resolve_cpu_allocation,
    serialize_task_domain,
)


DEFAULT_TRAIN_CANVAS_SIZE = (1000, 200)
DEFAULT_TRAIN_MAX_MODULE_SLOTS = 2048
DEFAULT_TRAIN_RELATIVE_INTERVAL_BINS = (0.08, 0.12, 0.16, 0.22, 0.30, 0.40)


def _read_config(config_path: Path) -> dict[str, Any]:
    with config_path.open(mode="r", encoding="utf-8") as file:
        raw = yaml.safe_load(file) or {}
    if not isinstance(raw, dict):
        raise ValueError("Training config must contain a mapping at the top level.")
    return raw


def _resolve_path(path_value: str | None, *, base_dir: Path) -> Path | None:
    if path_value is None:
        return None
    path = Path(path_value)
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _git_provenance() -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[1]
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                cwd=repo_root,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def _build_fab_model_config(domain: dict[str, Any]) -> Any:
    if "fab_models" in domain:
        return domain["fab_models"]

    base_model = copy.deepcopy(
        domain.get("fab_model", domain.get("base_fab_model", {}))
    )
    variants = domain.get("fab_model_variants", []) or []
    if not variants:
        return base_model or None
    if not isinstance(variants, list):
        raise ValueError("domain.fab_model_variants must be a list of mappings.")

    fab_models = [base_model]
    for variant in variants:
        if not isinstance(variant, dict):
            raise ValueError("Each domain.fab_model_variants entry must be a mapping.")
        model = copy.deepcopy(base_model)
        model.update(copy.deepcopy(variant))
        fab_models.append(model)
    return fab_models


def _build_env_kwargs(config: dict[str, Any], tasks) -> dict[str, Any]:
    schema = config.get("schema", {}) or {}
    environment = config.get("environment", {}) or {}
    scoring = config.get("scoring", {}) or {}
    canvas_size = tuple(schema.get("canvas_size", DEFAULT_TRAIN_CANVAS_SIZE))
    if len(canvas_size) != 2 or any(dimension <= 0 for dimension in canvas_size):
        raise ValueError(f"Invalid training canvas_size: {canvas_size}.")
    return {
        "package_size": tasks[0].package_size,
        "tasks": tasks,
        "patience": int(environment.get("patience", 5)),
        "result_dir": None,
        "estimate_workers": int(scoring.get("estimate_workers_per_env", 1)),
        "compute_ratio": float(environment.get("compute_ratio", 0.8)),
        "score_seed": int(scoring.get("score_seed", environment.get("score_seed", 0))),
        "score_seeds": [
            int(seed)
            for seed in scoring.get(
                "score_seeds", [scoring.get("score_seed", environment.get("score_seed", 0))]
            )
        ],
        "invalid_action_penalty": float(environment.get("invalid_action_penalty", -2.0)),
        "canvas_size": canvas_size,
        "max_module_slots": int(schema.get("max_module_slots", DEFAULT_TRAIN_MAX_MODULE_SLOTS)),
        "relative_interval_bins": tuple(
            float(value)
            for value in schema.get(
                "relative_interval_bins", DEFAULT_TRAIN_RELATIVE_INTERVAL_BINS
            )
        ),
        "base_seed": int(config.get("ppo", {}).get("seed", 42)),
        "record_design_history": False,
    }


def _resolve_model_paths(config: dict[str, Any], config_path: Path) -> tuple[Path, Path | None]:
    artifact = config.get("artifact", {}) or {}
    resume = config.get("resume", {}) or {}
    run_name = str(artifact.get("run_name", "layout-rl-1000x200"))
    model_root = _resolve_path(
        str(artifact.get("model_root", "./result/layout_models")),
        base_dir=config_path.parent,
    )
    paths = shared_layout_model_paths(model_root, run_name)
    output_model_path = _resolve_path(
        artifact.get("output_model_path"), base_dir=config_path.parent
    )
    if output_model_path is None:
        output_model_path = paths.model_path.resolve()
    initial_model_path = _resolve_path(
        resume.get("initial_model_path"), base_dir=config_path.parent
    )
    return output_model_path, initial_model_path


def _resolve_source_model(
    output_model_path: Path,
    initial_model_path: Path | None,
    resume_config: dict[str, Any],
) -> Path | None:
    output_exists = output_model_path.exists()
    continue_training = bool(resume_config.get("continue_training", True))
    overwrite = bool(resume_config.get("overwrite", False))

    if output_exists and initial_model_path is not None:
        if output_model_path.resolve() != initial_model_path.resolve():
            raise ValueError(
                "Both output model and initial_model_path exist but point to different "
                "checkpoints. Use a fresh run_name/output path to continue from an "
                "external checkpoint."
            )

    if output_exists:
        if continue_training:
            return output_model_path
        if not overwrite:
            raise FileExistsError(
                f"Model already exists at {output_model_path}; set resume.continue_training "
                "or resume.overwrite explicitly."
            )
        return initial_model_path

    return initial_model_path


def run_training(config_path: Path) -> None:
    config = _read_config(config_path)
    domain = config.get("domain", {}) or {}
    schema = config.get("schema", {}) or {}
    ppo = config.get("ppo", {}) or {}
    workers = config.get("workers", {}) or {}
    resume = config.get("resume", {}) or {}

    canvas_size = tuple(schema.get("canvas_size", DEFAULT_TRAIN_CANVAS_SIZE))
    tasks = build_layout_tasks(
        domain.get("package_shapes", [40, 60, 80, 100, 120, 140, 160]),
        _build_fab_model_config(domain),
        chiplet_length=int(
            domain.get(
                "chiplet_length",
                config.get("environment", {}).get("chiplet_length", 10),
            )
        ),
    )
    serialized_tasks = serialize_task_domain(tasks)

    output_model_path, initial_model_path = _resolve_model_paths(config, config_path)
    output_model_path.parent.mkdir(parents=True, exist_ok=True)
    run_manifest_path = output_model_path.with_name("run_manifest.json")
    source_model_path = _resolve_source_model(output_model_path, initial_model_path, resume)

    env_kwargs = _build_env_kwargs(config, tasks)
    reference_env = QuNoCEnv(**env_kwargs)

    prepare_subprocess_thread_limits()
    memory_per_worker_mb = workers.get("memory_per_worker_mb")
    memory_per_worker_bytes = (
        int(memory_per_worker_mb) * 1024 * 1024
        if memory_per_worker_mb is not None
        else None
    )
    requested_envs = workers.get("num_envs", "auto")
    if isinstance(requested_envs, str) and requested_envs.lower() != "auto":
        requested_envs = int(requested_envs)
    allocation = resolve_cpu_allocation(
        requested=requested_envs,
        minimum=int(workers.get("num_envs_min", 1)),
        maximum=(
            int(workers["num_envs_max"])
            if workers.get("num_envs_max") is not None
            else None
        ),
        reserve=int(workers.get("cpu_reserve", 2)),
        sample_seconds=float(workers.get("cpu_sample_seconds", 0.25)),
        memory_per_worker_bytes=memory_per_worker_bytes,
    )

    n_steps = int(ppo.get("n_steps", 16))
    batch_size = int(ppo.get("batch_size", 32))
    rollout_size = allocation.num_envs * n_steps
    if batch_size > rollout_size:
        raise ValueError(f"batch_size={batch_size} exceeds rollout size {rollout_size}.")
    if rollout_size % batch_size:
        logging.warning(
            "PPO rollout size %s is not divisible by batch size %s.",
            rollout_size,
            batch_size,
        )

    run_manifest = {
        "artifact_type": "qunoc-layout-training-run",
        "artifact_version": 1,
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path.resolve()),
        "output_model_path": str(output_model_path),
        "source_model_path": str(source_model_path) if source_model_path else None,
        "task_domain": serialized_tasks,
        "schema": {
            "canvas_size": list(canvas_size),
            "max_module_slots": int(env_kwargs["max_module_slots"]),
            "relative_interval_bins": list(env_kwargs["relative_interval_bins"]),
        },
        "ppo": copy.deepcopy(ppo),
        "cpu_allocation": allocation.to_dict(),
        "code_provenance": _git_provenance(),
    }
    write_json_atomic(run_manifest_path, run_manifest)

    training_env = reference_env
    if allocation.num_envs > 1:
        worker_env_kwargs = copy.deepcopy(env_kwargs)
        factories = [
            LayoutEnvFactory(
                env_kwargs=worker_env_kwargs,
                tasks=tasks,
                worker_index=worker_index,
                worker_count=allocation.num_envs,
                cpu_id=cpu_id,
                seed=int(ppo.get("seed", 42)),
            )
            for worker_index, cpu_id in enumerate(allocation.selected_cpu_ids)
        ]
        training_env = SubprocVecEnv(
            factories,
            start_method=str(workers.get("start_method", "forkserver")),
        )

    try:
        agent = LayoutAgent(
            training_env,
            pre_trained_model_path=str(source_model_path) if source_model_path else None,
            design_env=reference_env,
            seed=int(ppo.get("seed", 42)),
            learning_rate=float(ppo.get("learning_rate", 3e-4)),
            n_steps=n_steps,
            batch_size=batch_size,
            ent_coef=float(ppo.get("ent_coef", 0.01)),
            allow_legacy_model=False,
            training_config=copy.deepcopy(config),
            output_model_path=str(output_model_path),
            run_manifest_path=str(run_manifest_path),
            task_domain=serialized_tasks,
            resource_provenance={"cpu_allocation": allocation.to_dict()},
        )
        target_timesteps = int(ppo.get("target_timesteps", 32768))
        if source_model_path is None:
            agent.train(total_timesteps=target_timesteps)
        else:
            agent.train_until(target_timesteps)
        run_manifest.update(
            {
                "status": "completed",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "trained_timesteps": agent.trained_timesteps,
                "target_timesteps": target_timesteps,
            }
        )
    except Exception as error:
        run_manifest.update(
            {
                "status": "failed",
                "failed_at": datetime.now(timezone.utc).isoformat(),
                "error": str(error),
                "error_type": type(error).__name__,
            }
        )
        raise
    finally:
        if training_env is not reference_env:
            training_env.close()
        reference_env.close()
        write_json_atomic(run_manifest_path, run_manifest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a shared RL layout model only.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).with_name("train_layout_model.yaml")),
        help="Path to the layout training YAML config.",
    )
    args = parser.parse_args()
    run_training(Path(args.config))
