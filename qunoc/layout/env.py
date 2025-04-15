from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ..element.package import *


class QuantumChipletEnv(gym.Env):
    def __init__(
        self,
        package_size: tuple[int, int] = (50, 50),
        fabricate_model=FabricateModel(),
        patience=100,
    ):
        """
        Quantum Chiplet Environment
        package_size: size of the package to design

        observation space: The observation space is a 5-channel image of package size, where each channel represents:
        0. the type of the qubit (0: compute, 1: transfer, 2: empty)
        1-4. the boundary coordinate of the qubit belongs to.

        action space: The action space is a 5-dimensional vector, where each dimension represents:
        0. x coordinate of the qubit choosen to design whose area
        1. y coordinate of the qubit choosen to design whose area
        2. template choosen to design the area
        3. orientation choosen to design the area
        4. interval choosen to design the area
        """

        super(QuantumChipletEnv, self).__init__()
        self._package_size = package_size
        self._fabricate_model = fabricate_model
        self._patience = patience
        self.package = Package(self._package_size, self._fabricate_model)

        self._last_score = 0
        self._best_reward = -np.inf
        self._no_improve_steps = 0
        
        low_per_channel = np.array([0, 0, 0, 0, 0], dtype=np.int32)
        high_per_channel = np.array(
            [
                len(NodeType) - 1, # type of the cell
                self._package_size[0] - 1, # left boundary
                self._package_size[1] - 1, # top boundary
                self._package_size[0] - 1, # right boundary
                self._package_size[1] - 1, # bottom boundary
            ],
            dtype=np.int32,
        )
        self.observation_space = spaces.Box(
            np.broadcast_to(low_per_channel[:, None, None], (5, *self._package_size)),
            np.broadcast_to(high_per_channel[:, None, None], (5, *self._package_size)),
            (5, *self._package_size),
            np.int32,
        )
        
        self.action_space = spaces.MultiDiscrete(
            [
                self._package_size[0], # x coordinate of the qubit choosen to design whose area
                self._package_size[1], # y coordinate of the qubit choosen to design whose area
                len(DesignTemplate), # template choosen to design the area
                len(DesignOrientation), # orientation choosen to design the area
                max(self._package_size)//2, # interval choosen to design the area
            ]
        )

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed, options=options)
        self._last_score = 0
        self._best_reward = -np.inf
        self._no_improve_steps = 0
        self.package = Package(self._package_size, self._fabricate_model)
        return self._get_current_state(), {}

    def step(self, action):
        action = self.deserialize_action(action)
        if action is None:
            reward = -1e8
        else:
            new_compute_areas, new_transfer_lines = Package.design_module_template(
                **action, check_valid=False
            )
            self.package.compute_modules.remove(action["area"])
            self.package.compute_modules.extend(new_compute_areas)
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

    def deserialize_action(self, action: list):
        modified_action = {
            "area": self.package.get_physical_qubit_area((action[0], action[1])),
            "template": DesignTemplate(action[2]),
            "orient": DesignOrientation(action[3]),
            "interval": action[4],
        }
        if modified_action["area"] is not None and Package.is_valid_to_design(
            **modified_action
        ):
            return modified_action

    def _get_current_state(self):
        observation = np.zeros((5, *self._package_size), dtype=np.int32)
        for i in range(self._package_size[0]):
            for j in range(self._package_size[1]):
                area = self.package.get_physical_qubit_area((i, j))
                assert area is not None, "Area is None when calculate state."
                if area["type"] == NodeType.Compute:
                    observation[:, i, j] = [
                        1,
                        area["left"],
                        area["right"],
                        area["top"],
                        area["bottom"],
                    ]
                elif area["type"] == NodeType.Transfer:
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
