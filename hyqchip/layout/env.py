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
        self._package_size = package_size
        self._fabricate_model = fabricate_model
        self._patience = patience
        self.reset()
        self.action_space = spaces.Dict(
            {
                "pos": spaces.Tuple(
                    (
                        spaces.Discrete(self._package_size[0]),
                        spaces.Discrete(self._package_size[1]),
                    )
                ),
                "interval": spaces.Discrete(np.max(self._package_size)),
                "type": spaces.Discrete(3),
                "orient": spaces.Discrete(2),
            }
        )
        self.observation_space = spaces.Box(
            low=0, high=len(CellType) - 1, shape=(1, *package_size), dtype=np.int8
        )

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed, options=options)
        self._last_score = 0
        self._best_reward = -np.inf
        self._no_improve_steps = 0
        self.package = Package(self._package_size, self._fabricate_model)

    def step(self, action):
        action = self._modify_action(action)
        if action is None:
            reward = -np.inf
        else:
            score = self.estimate()
            reward = self._last_score - score
            self._last_score = score
        if reward > self._best_reward:
            self._best_reward = reward
            self._no_improve_steps = 0
        else:
            self._no_improve_steps += 1

        return (
            self.package.type_matrix,
            reward,
            self._no_improve_steps >= self._patience,
            False,
            {},
        )

    def estimate(self):
        return self.package.estimate_transfer()[0]

    def render(self, mode="human"):
        self.package.draw_qubits()

    def _modify_action(self, action):
        modified_action = {
            "interval": action["interval"] + 1,
            "type": action["type"],
            "orient": action["orient"],
            "area": self.package.locate_area(action["pos"]),
        }
        if modified_action["area"] is not None and Package.is_valid_to_design(
            **modified_action
        ):
            return modified_action


class QuantumChipletEnvV1(gym.Env):
    def __init__(
        self,
        package_size=(22, 11),
        fabricate_model=FabricateModel(),
        template="string",
        patience=100,
    ):
        super(QuantumChipletEnvV1, self).__init__()
        self.package_size = package_size
        self.fabricate_model = fabricate_model
        self.template = template

        self.package: Package = None
        self.state = np.zeros(
            self.package_size, dtype=np.int8
        )  # 0: Empty, 1: Compute, 2: Transfer
        self.score = 0

        self.action_space = spaces.Box(
            low=0, high=len(CellType) - 1, shape=self.package_size, dtype=np.int8
        )
        self.observation_space = spaces.Box(
            low=0, high=len(CellType) - 1, shape=(1, *package_size), dtype=np.int8
        )

        self.patience = patience
        self.best_reward = -np.inf
        self.no_improve_steps = 0

    def reset(self, seed=None, options=None):
        print("Resetting environment...")
        self.package = Package(
            self.package_size, self.fabricate_model, template=self.template
        )
        self.state = self.package.type_matrix
        self.best_reward = -np.inf
        self.no_improve_steps = 0
        return np.expand_dims(self.state, axis=0), {}

    def step(self, action):
        print("Taking action:")
        action = np.clip(np.round(action), 0, len(CellType) - 1).astype(np.int8)
        print(action)
        for i in range(self.package_size[0]):
            for j in range(self.package_size[1]):
                if action[i, j]:
                    self.state[i, j] = action[i, j]
        self.package.update_all_cells(self.state)
        self.package.complete_package()

        new_score, _, _ = self.package.estimate_transfer()
        reward = new_score - self.score
        self.score = new_score

        # Check if reward is improving, and stop if there's no improvement for `patience` steps
        if reward > self.best_reward:
            self.best_reward = reward
            self.no_improve_steps = 0
        else:
            self.no_improve_steps += 1
        done = self.no_improve_steps >= self.patience

        return np.expand_dims(self.state, axis=0), reward, done, False, {}

    def estimate(self):
        return self.package.estimate_transfer()[0]

    def render(self, mode="human"):
        self.package.draw_qubits()
