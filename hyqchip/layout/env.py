from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ..element.package import CellType, FabricateModel, Package


class QuantumChipletEnv(gym.Env):
    def __init__(
        self,
        package_size: tuple[int, int] = (22, 11),
        fabricate_model=FabricateModel(),
        patience=100,
    ):
        super(QuantumChipletEnv, self).__init__()
        self.package_size = package_size
        self._last_score = 0
        self._best_reward = -np.inf
        self._no_improve_steps = 0
        self._fabricate_model = fabricate_model
        self._patience = patience
        self.package = Package(self.package_size, self._fabricate_model)
        # pos(2), interval(1), template(1), orient(1)
        self.action_space = spaces.MultiDiscrete(
            [
                self.package_size[0],  # x 的取值范围: 0~21
                self.package_size[1],  # y 的取值范围: 0~10
                len(CellType),  # 类型选择（如0~3）
                3,  # 方向（0~2）
                2,  # 其他参数（0~1）
            ]
        )
        low_per_channel = np.array([0, 0, 0, 0, 0], dtype=np.int32)  # 5 个通道的最低值
        high_per_channel = np.array(
            [
                len(CellType) - 1,
                self.package_size[0] - 1,
                self.package_size[0] - 1,
                self.package_size[1] - 1,
                self.package_size[1] - 1,
            ],
            dtype=np.int32,
        )
        self.observation_space = spaces.Box(
            np.broadcast_to(low_per_channel[:, None, None], (5, *self.package_size)),
            np.broadcast_to(high_per_channel[:, None, None], (5, *self.package_size)),
            (5, *self.package_size),
            np.int32,
        )

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed, options=options)
        self._last_score = 0
        self._best_reward = -np.inf
        self._no_improve_steps = 0
        self.package = Package(self.package_size, self._fabricate_model)
        return self._get_current_state(), {}

    def step(self, action):
        action = self._modify_action(action)
        if action is None:
            reward = -1e8
        else:
            new_compute_areas, new_transfer_lines = Package.design_area(
                **action, check_valid=False
            )
            self.package.compute_areas.remove(action["area"])
            self.package.compute_areas.extend(new_compute_areas)
            self.package.transfer_lines.extend(new_transfer_lines)
            score = self.estimate()
            reward = self._last_score - score
            self._last_score = score
        if reward > self._best_reward:
            self._best_reward = reward
            self._no_improve_steps = 0
        else:
            self._no_improve_steps += 1

        return (
            self._get_current_state(),
            reward,
            self._no_improve_steps >= self._patience,
            False,
            {},
        )

    def estimate(self):
        self.package.complete_package()
        return self.package.estimate_transfer()[0]

    def render(self, mode="human"):
        self.package.draw_qubits()

    def _modify_action(self, action: list):
        modified_action = {
            "interval": action[2],
            "template": action[3],
            "orient": action[4],
            "area": self.package.get_physical_qubit_area((action[0], action[1])),
        }
        if modified_action["area"] is not None and Package.is_valid_to_design(
            **modified_action
        ):
            return modified_action

    def _get_current_state(self):
        observation = np.zeros((5, *self.package_size), dtype=np.int32)
        for i in range(self.package_size[0]):
            for j in range(self.package_size[1]):
                area = self.package.get_physical_qubit_area((i, j))
                assert area is not None, "Area is None when calculate state."
                if area["type"] == CellType.Compute:
                    observation[:, i, j] = [
                        1,
                        area["left"],
                        area["right"],
                        area["top"],
                        area["bottom"],
                    ]
                elif area["type"] == CellType.Transfer:
                    if area["axis"] == 0:
                        observation[:, i, j] = [
                            2,
                            area["start"],
                            area["end"],
                            area["pos"],
                            area["pos"],
                        ]
                    elif area["axis"] == 1:
                        observation[:, i, j] = [
                            2,
                            area["pos"],
                            area["pos"],
                            area["start"],
                            area["end"],
                        ]
                else:
                    pass
        return observation
