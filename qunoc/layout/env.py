import copy
import logging
import math
import random
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ..element.package import *
from .schema import POLICY_VERSION, SCHEMA_VERSION


class NoFeasibleDesignError(RuntimeError):
    """Raised when an episode ends before producing a feasible package."""


@dataclass(frozen=True)
class LayoutTask:
    """Immutable package-size and fabrication-parameter training task."""

    package_size: tuple[int, int]
    fab_model_kwargs: tuple[tuple[str, Any], ...] = ()
    name: str | None = None

    def __post_init__(self) -> None:
        package_size = tuple(map(int, self.package_size))
        if len(package_size) != 2 or any(dimension <= 0 for dimension in package_size):
            raise ValueError(f"Invalid package size: {self.package_size!r}")
        if isinstance(self.fab_model_kwargs, Mapping):
            fab_model_kwargs = tuple(
                (str(key), copy.deepcopy(value))
                for key, value in self.fab_model_kwargs.items()
            )
        else:
            fab_model_kwargs = tuple(
                (str(key), copy.deepcopy(value))
                for key, value in self.fab_model_kwargs
            )
        object.__setattr__(self, "package_size", package_size)
        object.__setattr__(self, "fab_model_kwargs", fab_model_kwargs)

    def build_fab_model(self) -> FabricateModel:
        return FabricateModel(**copy.deepcopy(dict(self.fab_model_kwargs)))


def _fabricate_model_kwargs(fab_model: FabricateModel) -> tuple[tuple[str, Any], ...]:
    return tuple(
        (key, copy.deepcopy(value))
        for key, value in vars(fab_model).items()
        if key not in {"intra_fide_params", "max_square_size"}
    )


def _normalize_layout_task(task: LayoutTask | Mapping[str, Any]) -> LayoutTask:
    if isinstance(task, LayoutTask):
        return task
    if isinstance(task, Mapping):
        try:
            fab_model_kwargs = task.get("fab_model_kwargs")
            if fab_model_kwargs is None:
                fab_model = task["fab_model"]
                if not isinstance(fab_model, FabricateModel):
                    raise TypeError("fab_model must be a FabricateModel instance.")
                fab_model_kwargs = _fabricate_model_kwargs(fab_model)
            return LayoutTask(
                package_size=task["package_size"],
                fab_model_kwargs=fab_model_kwargs,
                name=task.get("name"),
            )
        except KeyError as error:
            raise ValueError(
                "Each task mapping must define package_size and fab_model_kwargs "
                "(or legacy fab_model)."
            ) from error
    raise TypeError("tasks must contain LayoutTask instances or task mappings.")


class QuNoCEnv(gym.Env):
    """RL environment for top-down package layout design.

    The default fixed schema pads every package into a common canvas and exposes a
    fixed Discrete catalog of semantic actions. This allows one policy to be used
    across package sizes and fabrication configurations that fit the same schema.
    """

    metadata = {"render_modes": ["human"]}
    CONTEXT_FEATURES = (
        "package_width",
        "package_height",
        "aspect_ratio",
        "compute_ratio_target",
        "current_compute_ratio",
        "max_capacity",
        "max_square_size",
        "max_length",
        "inter_sparsity",
        "single_fide",
        "enhanced_fide",
        "inter_fide",
        "yield_5",
        "yield_decay",
        "geometry_square",
        "geometry_hexagon",
        "geometry_heavy_square",
        "geometry_heavy_hexagon",
    )

    def __init__(
        self,
        package_size: tuple[int, int] = (100, 100),
        fab_model: FabricateModel = FabricateModel(),
        result_dir: str = None,
        patience: int = 10,
        estimate_workers: int = 1,
        compute_ratio: float = 0.8,
        score_seed: int = 0,
        score_seeds: list[int] | tuple[int, ...] | None = None,
        invalid_action_penalty: float = -2.0,
        canvas_size: tuple[int, int] | None = None,
        max_module_slots: int = 256,
        relative_interval_bins: tuple[float, ...] = (0.08, 0.12, 0.16, 0.22, 0.30, 0.40),
        feasible_tolerance: float = 1e-9,
        package_sizes: Sequence[tuple[int, int]] | None = None,
        worker_index: int = 0,
        worker_count: int = 1,
        base_seed: int = 0,
        record_design_history: bool = True,
        tasks: Sequence[LayoutTask | Mapping[str, Any]] | None = None,
    ):
        super().__init__()
        initial_size = tuple(map(int, package_size))
        if tasks is not None and package_sizes is not None:
            raise ValueError("Pass either tasks or package_sizes, not both.")
        if tasks is None:
            sizes = package_sizes if package_sizes is not None else [initial_size]
            fab_model_kwargs = _fabricate_model_kwargs(fab_model)
            normalized_tasks = tuple(
                LayoutTask(package_size=size, fab_model_kwargs=fab_model_kwargs)
                for size in sizes
            )
        else:
            normalized_tasks = tuple(_normalize_layout_task(task) for task in tasks)
        if not normalized_tasks:
            raise ValueError("tasks must contain at least one layout task.")

        self.tasks = normalized_tasks
        self.package_sizes = tuple(task.package_size for task in self.tasks)
        self.package_size = self.tasks[0].package_size if tasks is not None else initial_size
        self.fab_model = self.tasks[0].build_fab_model() if tasks is not None else fab_model
        self.canvas_size = tuple(map(int, canvas_size or self.package_size))
        if len(self.canvas_size) != 2 or any(size <= 0 for size in self.canvas_size):
            raise ValueError("canvas_size must contain two positive dimensions.")
        for task in self.tasks:
            if any(
                package_dim > canvas_dim
                for package_dim, canvas_dim in zip(task.package_size, self.canvas_size)
            ):
                raise ValueError(
                    f"Package {task.package_size} does not fit canvas {self.canvas_size}."
                )
        self.worker_index = int(worker_index)
        self.worker_count = max(1, int(worker_count))
        self.base_seed = int(base_seed)
        self.episode_index = 0
        self.task_index = 0
        self.record_design_history = bool(record_design_history)
        if max_module_slots < 1:
            raise ValueError("max_module_slots must be positive.")
        if not relative_interval_bins or any(
            not 0.0 < value <= 0.5 for value in relative_interval_bins
        ):
            raise ValueError("relative_interval_bins must be in the interval (0, 0.5].")
        if not 0.0 <= compute_ratio <= 1.0:
            raise ValueError("compute_ratio must be between 0 and 1.")

        self.result_dir = result_dir
        self.patience = int(patience)
        self.estimate_workers = int(estimate_workers)
        self.compute_ratio_target = float(compute_ratio)
        self.score_seeds = tuple(
            int(seed) for seed in (score_seeds if score_seeds is not None else [score_seed])
        )
        if not self.score_seeds:
            raise ValueError("score_seeds must contain at least one seed.")
        self.score_seed = self.score_seeds[0]
        self.invalid_action_penalty = float(invalid_action_penalty)
        self.max_module_slots = int(max_module_slots)
        self.relative_interval_bins = tuple(float(value) for value in relative_interval_bins)
        self.feasible_tolerance = float(feasible_tolerance)
        self.schema_version = SCHEMA_VERSION
        self.policy_version = POLICY_VERSION
        self.interval_encoding = "relative-module-v1"

        layout_channels = 6
        self.actions_per_module = (
            len(DesignTemplate)
            * len(DesignOrientation)
            * len(self.relative_interval_bins)
        )
        self.stop_action = self.max_module_slots * self.actions_per_module
        self.action_space = spaces.Discrete(self.stop_action + 1)
        self.observation_space = spaces.Dict(
            {
                "layout": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(layout_channels, *self.canvas_size),
                    dtype=np.float32,
                ),
                "context": spaces.Box(
                    low=-1.0,
                    high=1.0,
                    shape=(len(self.CONTEXT_FEATURES),),
                    dtype=np.float32,
                ),
                "action_mask": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(self.action_space.n,),
                    dtype=np.float32,
                ),
            }
        )
        self._action_catalog: list[dict | None] | None = None
        self._action_mask: np.ndarray | None = None
        self._capacity_limited = False

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        options = options or {}
        forced_task_index = options.get("task_index")
        forced_size = options.get("package_size")
        if forced_task_index is not None:
            task_index = int(forced_task_index)
            if not 0 <= task_index < len(self.tasks):
                raise ValueError(
                    f"Task index {task_index} is outside this environment's task domain."
                )
            if forced_size is not None:
                requested_size = tuple(map(int, forced_size))
                if requested_size != self.tasks[task_index].package_size:
                    raise ValueError(
                        "task_index and package_size select different layout tasks."
                    )
        elif forced_size is not None:
            requested_size = tuple(map(int, forced_size))
            matching_indices = tuple(
                index
                for index, task in enumerate(self.tasks)
                if task.package_size == requested_size
            )
            if not matching_indices:
                raise ValueError(
                    f"Package size {requested_size} is outside this environment's task domain."
                )
            if len(matching_indices) > 1:
                raise ValueError(
                    f"Package size {requested_size} matches multiple fabrication tasks; "
                    "pass options['task_index'] to select one."
                )
            task_index = matching_indices[0]
        else:
            global_episode_index = self.worker_index + self.episode_index * self.worker_count
            round_index, round_offset = divmod(global_episode_index, len(self.tasks))
            task_index = self._task_order(round_index)[round_offset]
            self.episode_index += 1

        active_task = self.tasks[task_index]
        self.task_index = task_index
        self.package_size = active_task.package_size
        self.fab_model = active_task.build_fab_model()
        derived_seed = int(
            np.random.SeedSequence(
                [
                    self.base_seed,
                    self.worker_index,
                    self.episode_index,
                    task_index,
                    *self.package_size,
                ]
            ).generate_state(1)[0]
        )
        super().reset(seed=derived_seed if seed is None else seed, options=options)
        self.package = Package(
            self.package_size,
            self.fab_model,
            design_strategy="rl",
            result_dir=self.result_dir,
            record_design_history=self.record_design_history,
        )
        self.last_score, score_details = self.estimate_with_details()
        self.no_improve_steps = 0
        self.best_any_score = self.last_score
        self.best_any_package = copy.deepcopy(self.package)
        self.best_feasible_score = -math.inf
        self.best_feasible_package = None
        self.best_score = self.best_any_score
        self.best_package = self.best_any_package
        self._invalidate_action_catalog()
        self._update_best_packages(self.last_score)
        action_mask = self.get_action_mask()
        return self.get_current_state(), {
            "task_index": task_index,
            "task_name": active_task.name,
            "score": self.last_score,
            "score_details": score_details,
            "best_score": self.best_any_score,
            "best_feasible_score": self.best_feasible_score,
            "compute_ratio": self.get_compute_ratio(),
            "valid_action_count": self._ordinary_valid_action_count(action_mask),
            "action_mask": action_mask,
            "package_size": list(self.package_size),
            "worker_index": self.worker_index,
            "episode_index": self.episode_index,
            "seed": derived_seed if seed is None else int(seed),
        }

    def step(
        self, action: int | np.ndarray, record: bool = False
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        raw_action = int(np.asarray(action).item())
        action_mask = self.get_action_mask()
        action_valid = 0 <= raw_action < len(action_mask) and bool(action_mask[raw_action])
        decoded_action = self.deserialize_action(raw_action) if action_valid else None
        semantic_action = self._describe_action(decoded_action)
        step_outcome = "applied"
        termination_reason = None
        score_details = None

        if not action_valid:
            reward = self.invalid_action_penalty
            score = self.last_score
            self.no_improve_steps += 1
            step_outcome = "invalid_action"
        elif raw_action == self.stop_action:
            reward = 0.0
            score = self.last_score
            termination_reason = (
                "module_slot_capacity"
                if self._capacity_limited
                else "no_valid_actions"
            )
            step_outcome = "stop"
        else:
            self.package.design_module_top_down(**decoded_action, record=record)
            if len(self.package.compute_modules) > self.max_module_slots:
                raise AssertionError(
                    "Action-capacity filtering failed before package mutation."
                )
            self._invalidate_action_catalog()
            score, score_details = self.estimate_with_details()
            reward = score - self.last_score
            self.last_score = score
            self._update_best_packages(score)
            if reward > 0:
                self.no_improve_steps = 0
            else:
                self.no_improve_steps += 1

        compute_ratio = self.get_compute_ratio()
        next_action_mask = self.get_action_mask()
        valid_action_count = self._ordinary_valid_action_count(next_action_mask)
        if termination_reason is None:
            if self.is_feasible():
                termination_reason = "compute_ratio"
            elif valid_action_count == 0:
                termination_reason = (
                    "module_slot_capacity"
                    if self._capacity_limited
                    else "no_valid_actions"
                )
            elif self.no_improve_steps >= self.patience:
                termination_reason = "patience"

        active_task = self.tasks[self.task_index]
        info = {
            "task_index": self.task_index,
            "task_name": active_task.name,
            "raw_action": raw_action,
            "semantic_action": semantic_action,
            "action_valid": action_valid,
            "step_outcome": step_outcome,
            "score": score,
            "score_details": score_details,
            "best_score": self.best_any_score,
            "best_feasible_score": self.best_feasible_score,
            "compute_ratio": compute_ratio,
            "feasible": self.is_feasible(),
            "valid_action_count": valid_action_count,
            "action_mask": next_action_mask,
            "no_improve_steps": self.no_improve_steps,
            "termination_reason": termination_reason,
            "package_size": list(self.package_size),
            "worker_index": self.worker_index,
            "episode_index": self.episode_index,
        }
        return (
            self.get_current_state(),
            float(reward),
            termination_reason is not None,
            False,
            info,
        )

    def _task_order(self, round_index: int) -> tuple[int, ...]:
        """Return a reproducibly shuffled permutation for one task round."""
        order = list(range(len(self.tasks)))
        random.Random(self.base_seed + int(round_index)).shuffle(order)
        return tuple(order)

    def render(self, mode="human", filepath: str = None):
        try:
            fig, _ = self.package.draw_qubits()
            if mode == "human":
                fig.show()
            if filepath:
                fig.savefig(filepath, bbox_inches="tight")
        except Exception as error:
            logging.warning("Failed to draw qubits: %s", error)

    def estimate(self) -> float:
        return self.estimate_with_details()[0]

    def estimate_with_details(self) -> tuple[float, dict[str, Any]]:
        python_random_state = random.getstate()
        numpy_random_state = np.random.get_state()
        scores = []
        try:
            for seed in self.score_seeds:
                random.seed(seed)
                np.random.seed(seed)
                scores.append(
                    float(
                        self.package.estimate_score(
                            num_workers=self.estimate_workers
                        )
                    )
                )
        finally:
            random.setstate(python_random_state)
            np.random.set_state(numpy_random_state)
        return float(np.mean(scores)), {
            "seeds": list(self.score_seeds),
            "scores": scores,
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
            "min": float(np.min(scores)),
            "max": float(np.max(scores)),
        }

    def _decode_action_index(self, action_index: int) -> tuple[int, int, int, int]:
        if not 0 <= action_index < self.stop_action:
            raise ValueError(f"Invalid ordinary action index: {action_index}")
        module_slot, local_index = divmod(action_index, self.actions_per_module)
        template_orient_count = len(DesignOrientation) * len(self.relative_interval_bins)
        template_index, local_index = divmod(local_index, template_orient_count)
        orient_index, interval_index = divmod(
            local_index, len(self.relative_interval_bins)
        )
        return module_slot, template_index, orient_index, interval_index

    def encode_action_index(
        self,
        module_slot: int,
        template_index: int,
        orient_index: int,
        interval_index: int,
    ) -> int:
        return (
            module_slot * self.actions_per_module
            + template_index
            * len(DesignOrientation)
            * len(self.relative_interval_bins)
            + orient_index * len(self.relative_interval_bins)
            + interval_index
        )

    def _interval_for_module(
        self, module: ComputeModule, interval_index: int
    ) -> tuple[int, int]:
        module_width = module.right[0] - module.left[0] + 1
        module_height = module.bottom[0] - module.top[0] + 1
        ratio = self.relative_interval_bins[interval_index]
        return max(1, round(module_width * ratio)), max(1, round(module_height * ratio))

    def _build_action_catalog(self) -> None:
        catalog: list[dict | None] = [None] * int(self.action_space.n)
        mask = np.zeros(self.action_space.n, dtype=bool)
        modules = self._sorted_compute_modules()
        current_module_count = len(modules)
        capacity_limited = False
        for module_slot, module in enumerate(modules):
            seen_semantic_actions = set()
            for template in DesignTemplate:
                for orient in DesignOrientation:
                    for interval_index in range(len(self.relative_interval_bins)):
                        interval = self._interval_for_module(module, interval_index)
                        semantic_key = (template.value, orient.value, interval)
                        if semantic_key in seen_semantic_actions:
                            continue
                        action_dict = {
                            "old_module": module,
                            "template": template,
                            "orient": orient,
                            "interval": interval,
                        }
                        if not Package.is_valid_to_design(**action_dict):
                            continue
                        replacement_count = Package.count_modules_after_top_down_design(
                            **action_dict
                        )
                        if (
                            current_module_count - 1 + replacement_count
                            > self.max_module_slots
                        ):
                            capacity_limited = True
                            continue
                        seen_semantic_actions.add(semantic_key)
                        action_index = self.encode_action_index(
                            module_slot,
                            template.value,
                            orient.value,
                            interval_index,
                        )
                        catalog[action_index] = action_dict
                        mask[action_index] = True
        if not mask[: self.stop_action].any():
            mask[self.stop_action] = True
        self._action_catalog = catalog
        self._action_mask = mask
        self._capacity_limited = capacity_limited

    def _invalidate_action_catalog(self) -> None:
        self._action_catalog = None
        self._action_mask = None
        self._capacity_limited = False

    def get_action_mask(self) -> np.ndarray:
        if self._action_mask is None:
            self._build_action_catalog()
        return self._action_mask.copy()

    def deserialize_action(self, action: int | np.ndarray) -> dict | None:
        action_index = int(np.asarray(action).item())
        if action_index == self.stop_action:
            return None
        if not 0 <= action_index < self.action_space.n:
            return None
        if self._action_catalog is None:
            self._build_action_catalog()
        return self._action_catalog[action_index]

    @staticmethod
    def _describe_action(action: dict | None) -> dict | None:
        if action is None:
            return None
        return {
            "module": str(action["old_module"]),
            "template": action["template"].name,
            "orientation": action["orient"].name,
            "interval": list(action["interval"]),
        }

    def get_compute_ratio(self) -> float:
        designed_nodes = len(self.package.compute_nodes) + len(self.package.transfer_nodes)
        if designed_nodes == 0:
            return 1.0
        return len(self.package.compute_nodes) / designed_nodes

    def is_feasible(self) -> bool:
        return self.get_compute_ratio() <= (
            self.compute_ratio_target + self.feasible_tolerance
        )

    def _update_best_packages(self, score: float) -> None:
        if score > self.best_any_score:
            self.best_any_score = score
            self.best_any_package = copy.deepcopy(self.package)
        if self.is_feasible() and score > self.best_feasible_score:
            self.best_feasible_score = score
            self.best_feasible_package = copy.deepcopy(self.package)
        self.best_score = self.best_any_score
        self.best_package = (
            self.best_feasible_package
            if self.best_feasible_package is not None
            else self.best_any_package
        )

    def _sorted_compute_modules(self) -> list[ComputeModule]:
        modules = sorted(
            self.package.compute_modules,
            key=lambda module: (
                module.top[0],
                module.left[0],
                module.bottom[0],
                module.right[0],
            ),
        )
        if len(modules) > self.max_module_slots:
            raise RuntimeError(
                f"Layout has {len(modules)} compute modules, exceeding fixed schema "
                f"capacity {self.max_module_slots}."
            )
        return modules

    def _ordinary_valid_action_count(self, mask: np.ndarray | None = None) -> int:
        mask = self.get_action_mask() if mask is None else mask
        return int(mask[: self.stop_action].sum())

    def count_valid_actions(self) -> int:
        return self._ordinary_valid_action_count()

    def _get_context(self) -> np.ndarray:
        geometry = str(self.fab_model.compute_geometry).lower().replace(" ", "_")
        geometry_values = [
            float(geometry == name)
            for name in ("square", "hexagon", "heavy_square", "heavy_hexagon")
        ]
        width, height = self.package_size
        canvas_width, canvas_height = self.canvas_size
        values = [
            width / canvas_width,
            height / canvas_height,
            (width / height - 1.0) / max(width / height, 1.0),
            self.compute_ratio_target,
            self.get_compute_ratio(),
            min(1.0, self.fab_model.max_capacity / max(1, width * height)),
            min(1.0, self.fab_model.max_square_size / max(width, height)),
            min(1.0, self.fab_model.max_length / max(width, height)),
            min(1.0, self.fab_model.inter_sparsity / max(width, height)),
            self.fab_model.single_fide,
            self.fab_model.enhanced_fide,
            self.fab_model.inter_fide,
            self.fab_model.yield_5,
            math.tanh(self.fab_model.yield_decay),
            *geometry_values,
        ]
        return np.asarray(values, dtype=np.float32)

    def get_current_state(self) -> dict[str, np.ndarray]:
        observation = np.zeros((6, *self.canvas_size), dtype=np.float32)
        width_scale = max(1, self.package_size[0] - 1)
        height_scale = max(1, self.package_size[1] - 1)
        for node in self.package.package_graph.nodes:
            area = self.package.get_node_area(node)
            assert area is not None, "Area is None during layout design."
            node_type = self.package.package_graph.nodes[node]["type"].value
            if isinstance(area, ComputeModule):
                bounds = (
                    area.left[0] / width_scale,
                    area.right[0] / width_scale,
                    area.top[0] / height_scale,
                    area.bottom[0] / height_scale,
                )
            elif isinstance(area, TransferLine) and area.axis == 0:
                bounds = (
                    area.start[0] / width_scale,
                    area.end[0] / width_scale,
                    area.pos / height_scale,
                    area.pos / height_scale,
                )
            elif isinstance(area, TransferLine) and area.axis == 1:
                bounds = (
                    area.pos / width_scale,
                    area.pos / width_scale,
                    area.start[0] / height_scale,
                    area.end[0] / height_scale,
                )
            else:
                continue
            observation[:, node[0], node[1]] = [
                node_type / max(1, len(NodeType) - 1),
                *bounds,
                1.0,
            ]
        return {
            "layout": observation,
            "context": self._get_context(),
            "action_mask": self.get_action_mask().astype(np.float32),
        }
