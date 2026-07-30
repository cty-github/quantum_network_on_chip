import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import gymnasium as gym

from ..element.package import DesignOrientation, DesignTemplate


SCHEMA_VERSION = "fixed-canvas-candidate-v1"
POLICY_VERSION = "masked-candidate-ppo-min-compute-v2"


class IncompatibleLayoutModelError(ValueError):
    """Raised when a layout checkpoint cannot be used by an environment."""


def _space_schema(space: gym.Space) -> dict[str, Any]:
    if isinstance(space, gym.spaces.Box):
        return {
            "type": "Box",
            "shape": list(space.shape),
            "dtype": str(space.dtype),
        }
    if isinstance(space, gym.spaces.Discrete):
        return {"type": "Discrete", "n": int(space.n)}
    if isinstance(space, gym.spaces.MultiDiscrete):
        return {"type": "MultiDiscrete", "nvec": space.nvec.tolist()}
    if isinstance(space, gym.spaces.Dict):
        return {
            "type": "Dict",
            "spaces": {
                key: _space_schema(value) for key, value in sorted(space.spaces.items())
            },
        }
    raise TypeError(f"Unsupported Gym space for layout schema: {space!r}")


def canonical_json(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def schema_signature(schema: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(schema).encode("utf-8")).hexdigest()


def build_environment_schema(env) -> dict[str, Any]:
    schema = {
        "schema_version": getattr(env, "schema_version", "legacy-dynamic-v1"),
        "observation_space": _space_schema(env.observation_space),
        "action_space": _space_schema(env.action_space),
        "templates": [
            {"name": template.name, "value": int(template.value)}
            for template in DesignTemplate
        ],
        "orientations": [
            {"name": orient.name, "value": int(orient.value)}
            for orient in DesignOrientation
        ],
        "policy_version": getattr(env, "policy_version", "legacy-ppo-v1"),
        "interval_encoding": getattr(env, "interval_encoding", "absolute-chiplet-v1"),
        "context_features": list(getattr(env, "CONTEXT_FEATURES", ())),
        "layout_channels": [
            "node_type",
            "left_bound",
            "right_bound",
            "top_bound",
            "bottom_bound",
            "package_mask",
        ],
        "padding_alignment": "origin-top-left-v1",
        "action_index_encoding": "module-template-orientation-interval-v1",
        "stop_action_semantics": "legal-only-without-ordinary-actions-v1",
    }
    if hasattr(env, "canvas_size"):
        schema["canvas_size"] = list(env.canvas_size)
    if hasattr(env, "max_module_slots"):
        schema["max_module_slots"] = int(env.max_module_slots)
    if hasattr(env, "relative_interval_bins"):
        schema["relative_interval_bins"] = list(env.relative_interval_bins)
    return schema


def build_model_manifest(
    env,
    trained_timesteps: int,
    source_model_path: str | None,
    training_config: dict[str, Any] | None = None,
    *,
    output_model_path: str | None = None,
    run_manifest_path: str | None = None,
    task_domain: list[dict[str, Any]] | None = None,
    resource_provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    schema = build_environment_schema(env)
    return {
        "artifact_type": "qunoc-layout-model",
        "artifact_version": 2,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "schema": schema,
        "schema_signature": schema_signature(schema),
        "trained_timesteps": int(trained_timesteps),
        "source_model_path": source_model_path,
        "output_model_path": output_model_path,
        "run_manifest_path": run_manifest_path,
        "task_domain": task_domain or [],
        "resource_provenance": resource_provenance or {},
        "training_config": training_config or {},
    }


def manifest_path_for_model(model_path: str | os.PathLike[str]) -> Path:
    model_path = Path(model_path)
    if model_path.stem == "checkpoint":
        return model_path.with_name("checkpoint_manifest.json")
    return model_path.with_name("model_manifest.json")


def read_model_manifest(model_path: str | os.PathLike[str]) -> dict[str, Any] | None:
    manifest_path = manifest_path_for_model(model_path)
    if not manifest_path.exists():
        return None
    with manifest_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_model_manifest(
    model_path: str | os.PathLike[str], manifest: dict[str, Any]
) -> None:
    manifest_path = manifest_path_for_model(model_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2, sort_keys=True)
    os.replace(temporary_path, manifest_path)


def describe_schema_mismatch(
    expected: dict[str, Any], actual: dict[str, Any]
) -> list[str]:
    differences = []
    for key in sorted(set(expected) | set(actual)):
        if expected.get(key) != actual.get(key):
            differences.append(
                f"{key}: checkpoint={expected.get(key)!r}, target={actual.get(key)!r}"
            )
    return differences


def validate_model_manifest(model_path: str, env, allow_legacy: bool = True) -> None:
    manifest = read_model_manifest(model_path)
    if manifest is None:
        if allow_legacy:
            return
        manifest_name = manifest_path_for_model(model_path).name
        raise IncompatibleLayoutModelError(
            f"Layout model {model_path} has no {manifest_name}. "
            "Legacy models may only be loaded with explicit legacy compatibility."
        )

    target_schema = build_environment_schema(env)
    target_signature = schema_signature(target_schema)
    if manifest.get("schema_signature") == target_signature:
        return

    checkpoint_schema = manifest.get("schema", {})
    checkpoint_canvas = checkpoint_schema.get("canvas_size")
    target_canvas = target_schema.get("canvas_size")
    if checkpoint_canvas != target_canvas:
        raise IncompatibleLayoutModelError(
            f"Incompatible layout model {model_path}: checkpoint canvas is "
            f"{checkpoint_canvas}, but target canvas is {target_canvas}. "
            "Observation tensors and policy parameters are incompatible; "
            "train a new model with the target fixed canvas."
        )

    differences = describe_schema_mismatch(checkpoint_schema, target_schema)
    detail = "; ".join(differences) if differences else "unknown schema difference"
    raise IncompatibleLayoutModelError(
        f"Incompatible layout model {model_path}: {detail}. "
        "Use a checkpoint trained with the same RL schema or start a new model."
    )
