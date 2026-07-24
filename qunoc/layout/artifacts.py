import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SharedLayoutPaths:
    experiment_dir: Path
    run_name: str
    model_root: Path | None = None

    @property
    def run_dir(self) -> Path:
        if self.model_root is not None:
            return self.model_root / self.run_name
        return self.experiment_dir / "shared_rl" / self.run_name

    @property
    def model_path(self) -> Path:
        return self.run_dir / "model.zip"

    @property
    def run_manifest_path(self) -> Path:
        return self.run_dir / "run_manifest.json"

    def package_dir(self, package_size: tuple[int, int]) -> Path:
        width, height = package_size
        directory_name = f"{width}_rl" if width == height else f"{width}_{height}_rl"
        return self.experiment_dir / directory_name

    def package_path(self, package_size: tuple[int, int]) -> Path:
        return self.package_dir(package_size) / "package.pkl"


def _validate_run_name(run_name: str) -> str:
    run_name = str(run_name).strip()
    if not run_name or run_name in {".", ".."} or any(
        separator in run_name for separator in ("/", "\\")
    ):
        raise ValueError(f"Invalid shared RL run name: {run_name!r}")
    return run_name


def shared_layout_paths(
    experiment_dir: str | os.PathLike[str], run_name: str
) -> SharedLayoutPaths:
    return SharedLayoutPaths(Path(experiment_dir), _validate_run_name(run_name))


def shared_layout_model_paths(
    model_root: str | os.PathLike[str], run_name: str
) -> SharedLayoutPaths:
    return SharedLayoutPaths(Path(model_root), _validate_run_name(run_name), Path(model_root))


def write_json_atomic(path: str | os.PathLike[str], value: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open(mode="w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2, sort_keys=True)
    os.replace(temporary, target)


def read_json(path: str | os.PathLike[str]) -> dict[str, Any] | None:
    target = Path(path)
    if not target.exists():
        return None
    with target.open(mode="r", encoding="utf-8") as file:
        return json.load(file)
