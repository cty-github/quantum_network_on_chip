import copy
import logging
import math
import os
import time
from dataclasses import asdict, dataclass
from itertools import product
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from ..element.package import FabricateModel
from .env import LayoutTask, QuNoCEnv


@dataclass(frozen=True)
class CpuAllocation:
    requested: int | str
    allowed_cpu_ids: tuple[int, ...]
    selected_cpu_ids: tuple[int, ...]
    hard_capacity: int
    idle_cpu_equivalents: float
    reserve: int
    minimum: int
    maximum: int | None
    sample_seconds: float
    source: str
    available_memory_bytes: int | None = None
    memory_per_worker_bytes: int | None = None
    memory_capacity: int | None = None

    @property
    def num_envs(self) -> int:
        return len(self.selected_cpu_ids)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self) | {"num_envs": self.num_envs}


def _allowed_cpu_ids() -> tuple[int, ...]:
    try:
        cpu_ids = tuple(sorted(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        cpu_ids = tuple(range(os.cpu_count() or 1))
    return cpu_ids or (0,)


def _read_cpu_times() -> dict[int, tuple[int, int]]:
    cpu_times: dict[int, tuple[int, int]] = {}
    with open("/proc/stat", mode="r", encoding="utf-8") as file:
        for line in file:
            fields = line.split()
            if not fields or not fields[0].startswith("cpu") or fields[0] == "cpu":
                continue
            cpu_label = fields[0][3:]
            if not cpu_label.isdigit():
                continue
            values = [int(value) for value in fields[1:]]
            if len(values) < 4:
                continue
            # Linux accounts guest time inside user/nice already. Idle and iowait
            # are both available CPU time for this launch-time estimate.
            total = sum(values[:8])
            idle = values[3] + (values[4] if len(values) > 4 else 0)
            cpu_times[int(cpu_label)] = (total, idle)
    return cpu_times


def _sample_idle_fractions(
    cpu_ids: tuple[int, ...], sample_seconds: float
) -> dict[int, float]:
    try:
        before = _read_cpu_times()
        time.sleep(max(0.0, float(sample_seconds)))
        after = _read_cpu_times()
    except (OSError, ValueError) as error:
        logging.warning("Unable to sample per-CPU utilization: %s", error)
        return {cpu_id: 1.0 for cpu_id in cpu_ids}

    fractions: dict[int, float] = {}
    for cpu_id in cpu_ids:
        if cpu_id not in before or cpu_id not in after:
            fractions[cpu_id] = 1.0
            continue
        total_delta = after[cpu_id][0] - before[cpu_id][0]
        idle_delta = after[cpu_id][1] - before[cpu_id][1]
        fractions[cpu_id] = (
            min(1.0, max(0.0, idle_delta / total_delta))
            if total_delta > 0
            else 1.0
        )
    return fractions


def _cgroup_cpu_capacity() -> int | None:
    cpu_max_path = "/sys/fs/cgroup/cpu.max"
    try:
        with open(cpu_max_path, mode="r", encoding="utf-8") as file:
            quota_value, period_value = file.read().split()[:2]
        if quota_value == "max":
            return None
        quota = int(quota_value)
        period = int(period_value)
        if quota <= 0 or period <= 0:
            return None
        return max(1, math.floor(quota / period))
    except (OSError, ValueError, IndexError):
        return None


def _slurm_cpu_capacity() -> int | None:
    value = os.environ.get("SLURM_CPUS_PER_TASK")
    if value is None:
        return None
    try:
        capacity = int(value)
    except ValueError:
        logging.warning("Ignore invalid SLURM_CPUS_PER_TASK=%r", value)
        return None
    return capacity if capacity > 0 else None


def _available_memory_bytes() -> int | None:
    try:
        with open("/proc/meminfo", mode="r", encoding="utf-8") as file:
            for line in file:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


def _cgroup_memory_available_bytes() -> int | None:
    try:
        with open("/sys/fs/cgroup/memory.max", mode="r", encoding="utf-8") as file:
            limit_value = file.read().strip()
        if limit_value == "max":
            return None
        with open("/sys/fs/cgroup/memory.current", mode="r", encoding="utf-8") as file:
            current_value = int(file.read().strip())
        return max(0, int(limit_value) - current_value)
    except (OSError, ValueError):
        return None


def _effective_available_memory_bytes() -> int | None:
    capacities = tuple(
        value
        for value in (_available_memory_bytes(), _cgroup_memory_available_bytes())
        if value is not None
    )
    return min(capacities) if capacities else None


def resolve_cpu_allocation(
    requested: int | str = "auto",
    minimum: int = 1,
    maximum: int | None = None,
    reserve: int = 1,
    sample_seconds: float = 0.25,
    memory_per_worker_bytes: int | None = None,
) -> CpuAllocation:
    allowed_cpu_ids = _allowed_cpu_ids()
    capacities = [len(allowed_cpu_ids)]
    sources = [f"affinity={len(allowed_cpu_ids)}"]
    cgroup_capacity = _cgroup_cpu_capacity()
    if cgroup_capacity is not None:
        capacities.append(cgroup_capacity)
        sources.append(f"cgroup={cgroup_capacity}")
    slurm_capacity = _slurm_cpu_capacity()
    if slurm_capacity is not None:
        capacities.append(slurm_capacity)
        sources.append(f"slurm={slurm_capacity}")
    hard_capacity = max(1, min(capacities))
    available_memory_bytes = None
    memory_capacity = None
    if memory_per_worker_bytes is not None:
        memory_per_worker_bytes = int(memory_per_worker_bytes)
        if memory_per_worker_bytes <= 0:
            raise ValueError("memory_per_worker_bytes must be positive when provided.")
        available_memory_bytes = _effective_available_memory_bytes()
        if available_memory_bytes is not None:
            memory_capacity = available_memory_bytes // memory_per_worker_bytes
            if memory_capacity < 1:
                raise ValueError(
                    "Available memory is below the configured per-layout-worker "
                    f"requirement ({available_memory_bytes} < {memory_per_worker_bytes} bytes)."
                )
            hard_capacity = min(hard_capacity, memory_capacity)
            sources.append(
                f"memory={memory_capacity} workers "
                f"({available_memory_bytes} bytes available)"
            )

    minimum = max(1, int(minimum))
    reserve = max(0, int(reserve))
    if maximum is not None:
        maximum = max(1, int(maximum))

    if isinstance(requested, str):
        if requested.lower() != "auto":
            raise ValueError("rl_num_envs must be a positive integer or 'auto'.")
        idle_fractions = _sample_idle_fractions(allowed_cpu_ids, sample_seconds)
        ranked_cpu_ids = tuple(
            sorted(
                allowed_cpu_ids,
                key=lambda cpu_id: (-idle_fractions[cpu_id], cpu_id),
            )
        )
        idle_equivalents = float(sum(idle_fractions.values()))
        available_after_reserve = max(1, hard_capacity - reserve)
        effective_maximum = available_after_reserve
        if maximum is not None:
            effective_maximum = min(effective_maximum, maximum)
        effective_minimum = min(minimum, effective_maximum)
        idle_candidate = max(1, math.floor(idle_equivalents) - reserve)
        num_envs = min(
            effective_maximum,
            max(effective_minimum, idle_candidate),
        )
    else:
        ranked_cpu_ids = allowed_cpu_ids
        idle_equivalents = float(len(allowed_cpu_ids))
        num_envs = int(requested)
        if num_envs < 1:
            raise ValueError("rl_num_envs must be at least 1.")
        if num_envs > hard_capacity:
            raise ValueError(
                f"Requested rl_num_envs={num_envs} exceeds effective CPU "
                f"capacity {hard_capacity} ({', '.join(sources)})."
            )

    selected_cpu_ids = ranked_cpu_ids[:num_envs]
    allocation = CpuAllocation(
        requested=requested,
        allowed_cpu_ids=allowed_cpu_ids,
        selected_cpu_ids=selected_cpu_ids,
        hard_capacity=hard_capacity,
        idle_cpu_equivalents=idle_equivalents,
        reserve=reserve,
        minimum=minimum,
        maximum=maximum,
        sample_seconds=float(sample_seconds),
        source=", ".join(sources),
        available_memory_bytes=available_memory_bytes,
        memory_per_worker_bytes=memory_per_worker_bytes,
        memory_capacity=memory_capacity,
    )
    logging.info("Resolved RL CPU allocation: %s", allocation.to_dict())
    print(
        "RL CPU allocation: "
        f"requested={requested} hard_capacity={hard_capacity} "
        f"idle_equivalents={idle_equivalents:.1f} reserve={reserve} "
        f"num_envs={allocation.num_envs}",
        flush=True,
    )
    return allocation


def build_package_sizes(
    values: Sequence[int | Sequence[int]],
) -> tuple[tuple[int, int], ...]:
    """Normalize square or explicit package sizes for shared layout training."""
    package_sizes = []
    for value in values:
        if isinstance(value, (int, np.integer)):
            package_size = (int(value), int(value))
        else:
            package_size = tuple(map(int, value))
        if len(package_size) != 2 or any(dimension <= 0 for dimension in package_size):
            raise ValueError(f"Invalid package size: {value!r}")
        package_sizes.append(package_size)
    if not package_sizes:
        raise ValueError("package sizes must contain at least one value.")
    if len(set(package_sizes)) != len(package_sizes):
        raise ValueError("package sizes must not contain duplicates.")
    return tuple(package_sizes)


def build_fab_models(
    values: Sequence[FabricateModel | Mapping[str, Any]]
    | FabricateModel
    | Mapping[str, Any]
    | None = None,
    *,
    chiplet_length: int = 10,
) -> tuple[FabricateModel, ...]:
    """Build fabrication models with the established chiplet-size defaults."""
    if values is None:
        model_values: Sequence[FabricateModel | Mapping[str, Any]] = ({},)
    elif isinstance(values, (FabricateModel, Mapping)):
        model_values = (values,)
    else:
        model_values = values
    if not model_values:
        raise ValueError("fabrication models must contain at least one value.")

    chiplet_length = int(chiplet_length)
    if chiplet_length <= 0:
        raise ValueError("chiplet_length must be positive.")
    models = []
    for value in model_values:
        if isinstance(value, FabricateModel):
            models.append(value)
            continue
        if not isinstance(value, Mapping):
            raise TypeError(
                "fabrication models must be FabricateModel instances or mappings."
            )
        kwargs = copy.deepcopy(dict(value))
        kwargs.setdefault("max_capacity", chiplet_length**2)
        kwargs.setdefault("max_length", int(0.8 * chiplet_length**2))
        models.append(FabricateModel(**kwargs))
    return tuple(models)


def build_layout_tasks(
    package_sizes: Sequence[int | Sequence[int]],
    fabricate_models: Sequence[FabricateModel | Mapping[str, Any]]
    | FabricateModel
    | Mapping[str, Any]
    | None = None,
    *,
    chiplet_length: int = 10,
) -> tuple[LayoutTask, ...]:
    """Build the immutable size-by-fabrication Cartesian task domain."""
    sizes = build_package_sizes(package_sizes)
    models = build_fab_models(fabricate_models, chiplet_length=chiplet_length)
    return tuple(
        LayoutTask(
            package_size=package_size,
            fab_model_kwargs=tuple(
                (key, copy.deepcopy(value))
                for key, value in vars(fab_model).items()
                if key not in {"intra_fide_params", "max_square_size"}
            ),
        )
        for package_size, fab_model in product(sizes, models)
    )


def serialize_task_domain(tasks: Sequence[LayoutTask]) -> list[dict[str, Any]]:
    return [
        {
            "task_id": task.name,
            "package_size": list(task.package_size),
            "fab_model": copy.deepcopy(dict(task.fab_model_kwargs)),
        }
        for task in tasks
    ]


def prepare_subprocess_thread_limits() -> None:
    for variable in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "BLIS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ.setdefault(variable, "1")


def configure_layout_worker(cpu_id: int | None, seed: int) -> None:
    if cpu_id is not None and hasattr(os, "sched_setaffinity"):
        try:
            os.sched_setaffinity(0, {int(cpu_id)})
        except OSError as error:
            logging.warning("Unable to pin layout worker to CPU %s: %s", cpu_id, error)
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # Inter-op threads can only be configured before parallel work starts.
        pass
    np.random.seed(int(seed))


@dataclass
class LayoutEnvFactory:
    env_kwargs: dict[str, Any]
    package_sizes: tuple[tuple[int, int], ...] | None = None
    worker_index: int = 0
    worker_count: int = 1
    cpu_id: int | None = None
    seed: int = 0
    tasks: tuple[LayoutTask, ...] | None = None

    def __call__(self) -> QuNoCEnv:
        worker_seed = int(self.seed) + self.worker_index
        configure_layout_worker(self.cpu_id, worker_seed)
        kwargs = copy.deepcopy(self.env_kwargs)
        if self.tasks is not None:
            kwargs.pop("package_sizes", None)
            kwargs["tasks"] = self.tasks
        elif self.package_sizes is not None:
            kwargs["package_sizes"] = self.package_sizes
        kwargs.update(
            {
                "worker_index": self.worker_index,
                "worker_count": self.worker_count,
                "base_seed": int(self.seed),
                "record_design_history": False,
                "estimate_workers": 1,
            }
        )
        return QuNoCEnv(**kwargs)
