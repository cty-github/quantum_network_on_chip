import argparse
import contextlib
import copy
import csv
import logging
import math
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.append(str(Path(__file__).resolve().parents[1]))

import yaml
import numpy as np
from stable_baselines3.common.vec_env import SubprocVecEnv

from qunoc.layout.agent import LayoutAgent
from qunoc.layout.artifacts import shared_layout_model_paths, write_json_atomic
from qunoc.layout.env import QuNoCEnv
from qunoc.layout.schema import manifest_path_for_model
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


class _TeeStream:
    def __init__(self, *streams):
        self._streams = streams

    @property
    def encoding(self):
        return getattr(self._streams[0], "encoding", "utf-8")

    @property
    def errors(self):
        return getattr(self._streams[0], "errors", "replace")

    def write(self, data):
        for stream in self._streams:
            stream.write(data)
        return len(data)

    def flush(self):
        for stream in self._streams:
            stream.flush()

    def isatty(self):
        return any(getattr(stream, "isatty", lambda: False)() for stream in self._streams)


@contextlib.contextmanager
def _tee_training_output(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", buffering=1) as log_file:
        with contextlib.redirect_stdout(_TeeStream(sys.stdout, log_file)):
            with contextlib.redirect_stderr(_TeeStream(sys.stderr, log_file)):
                yield


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


def _package_compute_ratio(package) -> float:
    designed_nodes = len(package.compute_nodes) + len(package.transfer_nodes)
    if designed_nodes == 0:
        return 1.0
    return len(package.compute_nodes) / designed_nodes


def _estimate_package_metrics(
    package,
    *,
    score_seed: int,
    estimate_workers: int,
) -> tuple[float, float, float, float]:
    python_random_state = random.getstate()
    numpy_random_state = np.random.get_state()
    random.seed(score_seed)
    np.random.seed(score_seed)
    try:
        score = float(package.estimate_score(num_workers=estimate_workers))
        fide_avg, fide_std = package.estimate_transfer(num_workers=estimate_workers)
        compute_ratio = _package_compute_ratio(package)
        return float(score), float(fide_avg), float(fide_std), float(compute_ratio)
    finally:
        random.setstate(python_random_state)
        np.random.set_state(numpy_random_state)


def _write_training_domain_inference_csv(
    agent: LayoutAgent,
    env_kwargs: dict[str, Any],
    tasks,
    output_path: Path,
    *,
    model_path: Path,
    artifact_role: str,
    trained_timesteps: int,
    inference_seed: int,
) -> None:
    fieldnames = [
        "artifact_role",
        "model_path",
        "trained_timesteps",
        "evaluated_at",
        "task_index",
        "task_id",
        "package_width",
        "package_height",
        "compute_geometry",
        "inter_sparsity",
        "inter_fide",
        "score",
        "fide_avg",
        "fide_std",
        "compute_ratio",
        "feasible",
        "termination_reason",
        "error",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not output_path.exists()
    evaluated_at = datetime.now(timezone.utc).isoformat()
    score_seed = int(env_kwargs.get("score_seed", 0))
    estimate_workers = int(env_kwargs.get("estimate_workers", 1))
    rows = []

    for task_index, task in enumerate(tasks):
        task_env = QuNoCEnv(**env_kwargs)
        task_seed = int(inference_seed) + task_index
        task_row = {
            "artifact_role": artifact_role,
            "model_path": str(model_path),
            "trained_timesteps": int(trained_timesteps),
            "evaluated_at": evaluated_at,
            "task_index": task_index,
            "task_id": task.name or f"task_{task_index}",
            "package_width": task.package_size[0],
            "package_height": task.package_size[1],
            "compute_geometry": dict(task.fab_model_kwargs).get("compute_geometry"),
            "inter_sparsity": dict(task.fab_model_kwargs).get("inter_sparsity"),
            "inter_fide": dict(task.fab_model_kwargs).get("inter_fide"),
            "score": math.nan,
            "fide_avg": math.nan,
            "fide_std": math.nan,
            "compute_ratio": math.nan,
            "feasible": False,
            "termination_reason": None,
            "error": None,
        }
        try:
            state, _ = task_env.reset(seed=task_seed, options={"task_index": task_index})
            done = False
            final_info = {}
            while not done:
                action, _ = agent.model.predict(state, deterministic=True)
                state, _, terminated, truncated, final_info = task_env.step(
                    action,
                    record=False,
                )
                done = bool(terminated or truncated)

            package = task_env.best_feasible_package or task_env.best_any_package
            score, fide_avg, fide_std, compute_ratio = _estimate_package_metrics(
                package,
                score_seed=score_seed,
                estimate_workers=estimate_workers,
            )
            task_row.update(
                {
                    "score": score,
                    "fide_avg": fide_avg,
                    "fide_std": fide_std,
                    "compute_ratio": compute_ratio,
                    "feasible": task_env.best_feasible_package is not None,
                    "termination_reason": final_info.get("termination_reason"),
                }
            )
        except Exception as error:
            task_row["error"] = f"{type(error).__name__}: {error}"
        finally:
            task_env.close()
        rows.append(task_row)

    with output_path.open(mode="a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def _resolve_source_model(
    output_model_path: Path,
    initial_model_path: Path | None,
    resume_config: dict[str, Any],
) -> tuple[Path | None, bool]:
    output_exists = output_model_path.exists()
    checkpoint_path = output_model_path.with_name("checkpoint.zip")
    checkpoint_exists = checkpoint_path.exists()
    continue_training = bool(resume_config.get("continue_training", True))
    resume_from_checkpoint = bool(resume_config.get("resume_from_checkpoint", True))
    overwrite = bool(resume_config.get("overwrite", False))

    if checkpoint_exists and resume_from_checkpoint:
        if initial_model_path is not None and checkpoint_path.resolve() != initial_model_path.resolve():
            raise ValueError(
                "Both checkpoint.zip and initial_model_path exist but point to different "
                "checkpoints. Remove checkpoint.zip or unset resume.initial_model_path."
            )
        if output_exists and not continue_training:
            raise FileExistsError(
                f"Model already exists at {output_model_path}; set resume.continue_training "
                "or resume.overwrite explicitly."
            )
        return checkpoint_path, True

    if output_exists and initial_model_path is not None:
        if output_model_path.resolve() != initial_model_path.resolve():
            raise ValueError(
                "Both output model and initial_model_path exist but point to different "
                "checkpoints. Use a fresh run_name/output path to continue from an "
                "external checkpoint."
            )

    if output_exists:
        if continue_training:
            return output_model_path, False
        if not overwrite:
            raise FileExistsError(
                f"Model already exists at {output_model_path}; set resume.continue_training "
                "or resume.overwrite explicitly."
            )
        return initial_model_path, False

    return initial_model_path, False


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
    training_log_path = output_model_path.with_name("training.log")
    with _tee_training_output(training_log_path):
        print(f"Training log: {training_log_path}", flush=True)
        run_manifest_path = output_model_path.with_name("run_manifest.json")
        checkpoint_path = output_model_path.with_name("checkpoint.zip")
        checkpoint_manifest_path = manifest_path_for_model(checkpoint_path)
        training_domain_inference_csv_path = output_model_path.with_name(
            "training_domain_inference.csv"
        )
        checkpoint_interval = int(
            resume.get(
                "checkpoint_interval_timesteps",
                ppo.get("checkpoint_interval_timesteps", 4096),
            )
        )
        checkpoint_enabled = bool(resume.get("checkpoint_enabled", True))
        if checkpoint_enabled and checkpoint_interval <= 0:
            raise ValueError("resume.checkpoint_interval_timesteps must be positive.")
        source_model_path, source_is_checkpoint = _resolve_source_model(
            output_model_path,
            initial_model_path,
            resume,
        )

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
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_manifest_path": str(checkpoint_manifest_path),
            "checkpoint_interval_timesteps": checkpoint_interval if checkpoint_enabled else None,
            "training_log_path": str(training_log_path),
            "training_domain_inference_csv_path": str(training_domain_inference_csv_path),
            "source_model_path": str(source_model_path) if source_model_path else None,
            "source_is_checkpoint": source_is_checkpoint,
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

            inference_seed = int(ppo.get("seed", 42)) + 1000

            def record_model_inference(
                model_file: Path,
                artifact_role: str,
                trained_timesteps: int,
            ) -> None:
                _write_training_domain_inference_csv(
                    agent,
                    env_kwargs,
                    tasks,
                    training_domain_inference_csv_path,
                    model_path=model_file,
                    artifact_role=artifact_role,
                    trained_timesteps=trained_timesteps,
                    inference_seed=inference_seed,
                )
                run_manifest.update(
                    {
                        "last_training_domain_inference_csv_path": str(
                            training_domain_inference_csv_path
                        ),
                        "last_training_domain_inference_at": datetime.now(
                            timezone.utc
                        ).isoformat(),
                    }
                )
                write_json_atomic(run_manifest_path, run_manifest)

            def record_checkpoint(checkpoint_file: Path, checkpoint_timesteps: int) -> None:
                agent.write_checkpoint_manifest(checkpoint_file, checkpoint_timesteps)
                run_manifest.update(
                    {
                        "last_checkpoint_path": str(checkpoint_file),
                        "last_checkpoint_manifest_path": str(
                            manifest_path_for_model(checkpoint_file)
                        ),
                        "last_checkpoint_timesteps": int(checkpoint_timesteps),
                        "last_checkpoint_at": datetime.now(timezone.utc).isoformat(),
                    }
                )
                record_model_inference(
                    checkpoint_file,
                    "checkpoint",
                    checkpoint_timesteps,
                )

            target_timesteps = int(ppo.get("target_timesteps", 32768))
            if source_model_path is None:
                agent.train(
                    total_timesteps=target_timesteps,
                    checkpoint_path=checkpoint_path if checkpoint_enabled else None,
                    checkpoint_interval=checkpoint_interval if checkpoint_enabled else None,
                    checkpoint_on_save=record_checkpoint if checkpoint_enabled else None,
                )
            else:
                agent.train_until(
                    target_timesteps,
                    checkpoint_path=checkpoint_path if checkpoint_enabled else None,
                    checkpoint_interval=checkpoint_interval if checkpoint_enabled else None,
                    checkpoint_on_save=record_checkpoint if checkpoint_enabled else None,
                )
            record_model_inference(
                output_model_path,
                "model",
                agent.trained_timesteps,
            )
            run_manifest.update(
                {
                    "status": "completed",
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                    "trained_timesteps": agent.trained_timesteps,
                    "target_timesteps": target_timesteps,
                }
            )
            if checkpoint_enabled:
                checkpoint_path.unlink(missing_ok=True)
                checkpoint_manifest_path.unlink(missing_ok=True)
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
