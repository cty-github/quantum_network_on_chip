from typing import Any

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from ..element.package import *


class QuantumChipletEnv(gym.Env):
    """
    Environment for designing quantum chiplets.

    Attributes:
        package_size (tuple[int, int]): Size of the package to design.
        fab_model (FabricateModel): Fabrication model for the quantum chiplet.
        patience (int): Number of steps to wait before stopping if no improvement.

        package (Package): The package to perform training.
        last_score (float): The last score of the package.
        no_improve_steps (int): Number of steps without improvement.

        observation space (spaces.Box): The observation space is a 5-channel image of package size, where each channel represents:
            0. the type of the qubit (0: compute, 1: transfer, 2: empty)
            1-4. the boundary coordinate of the qubit belongs to.

        action space (spaces.MultiDiscrete): The action space is a 5-dimensional vector, where each dimension represents:
            0. x coordinate of the qubit choosen to design whose area
            1. y coordinate of the qubit choosen to design whose area
            2. template choosen to design the area
            3. orientation choosen to design the area
            4. interval choosen to design the area
    """
    
    def __init__(self, 
        package_size: tuple[int, int] = (50, 50), 
        fab_model: FabricateModel = FabricateModel(), 
        patience: int = 100, 
    ):
        """
        Initialize the QuantumChipletEnv environment.

        Args:
            package_size (tuple[int, int]): Size of the package to design.
            fab_model (FabricateModel): Fabrication model for the quantum chiplet.
            patience (int): Number of steps to wait before stopping if no improvement.
        """
        super(QuantumChipletEnv, self).__init__()
        self.package_size = package_size
        self.fab_model = fab_model
        self.patience = patience
        
        low_per_channel = np.array([0, 0, 0, 0, 0], dtype=np.int32)
        high_per_channel = np.array(
            [
                len(NodeType) - 1, # type of the cell
                self.package_size[0] - 1, # left boundary
                self.package_size[1] - 1, # top boundary
                self.package_size[0] - 1, # right boundary
                self.package_size[1] - 1, # bottom boundary
            ],
            dtype=np.int32,
        )
        self.observation_space = spaces.Box(
            np.broadcast_to(low_per_channel[:, None, None], (5, *self.package_size)),
            np.broadcast_to(high_per_channel[:, None, None], (5, *self.package_size)),
            (5, *self.package_size),
            np.int32,
        )
        self.action_space = spaces.MultiDiscrete(
            [
                self.package_size[0], # x coordinate of the qubit choosen to design whose area
                self.package_size[1], # y coordinate of the qubit choosen to design whose area
                len(DesignTemplate), # template choosen to design the area
                len(DesignOrientation), # orientation choosen to design the area
                (max(self.package_size) // 2 - 1) // self.fab_model.max_square_size + 1, # number of interval choosen to design the area
            ]
        )

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed, options=options)
        self.package = Package(self.package_size, self.fab_model, design_strategy="rl")
        self.last_score = self.estimate()
        self.no_improve_steps = 0
        return self.get_current_state(), {}

    def step(self, action):
        action = self.deserialize_action(action)
        if action is None:
            reward = -1e8
        else:
            self.package.design_module_top_down(**action)
            score = self.estimate()
            reward = score - self.last_score
            self.last_score = score
        if reward > 0:
            self.no_improve_steps = 0
        else:
            self.no_improve_steps += 1

        return (
            self.get_current_state(),
            reward,
            self.no_improve_steps >= self.patience,
            False,
            {},
        )

    def render(self, mode="human"):
        self.package.draw_qubits()

    def estimate(self) -> float:
        """
        Estimate the efficiency of the package.

        Returns:
            float: The evaluation score of the package.
        """
        self.package.complete_package()
        return self.package.estimate_transfer()

    def deserialize_action(self, action_vector: list) -> dict | None:
        """
        Deserialize the action vector into a dictionary of parameters.

        Args:
            action_vector (list): The action vector to deserialize.
        
        Returns:
            dict | None: The deserialized action dictionary or None if the action is invalid.
        """
        action_dict = {
            "module": self.package.get_physical_qubit_area((action_vector[0], action_vector[1])),
            "template": DesignTemplate(action_vector[2]),
            "orient": DesignOrientation(action_vector[3]),
            "interval": 1 + action_vector[4] * self.fab_model.max_square_size,
        }
        if action_dict["module"].type == NodeType.Compute and Package.is_valid_to_design(**action_dict):
            return action_dict
        else:
            return None

    def get_current_state(self) -> np.ndarray:
        """
        Get the current state of the environment.

        Returns:
            np.ndarray: The current state of the environment, represented as a 5-channel image of package size.
        """
        observation = np.zeros((5, *self.package_size), dtype=np.int32)
        for node in self.package.package_graph.nodes:
            area = self.package.get_physical_qubit_area(node)
            assert area is not None, "Area is None during layout design."
            if isinstance(area, ComputeModule):
                observation[:, node[0], node[1]] = [
                    area.type.value,
                    area.left,
                    area.right,
                    area.top,
                    area.bottom,
                ]
            elif isinstance(area, TransferLine):
                if area.axis == 0:
                    observation[:, node[0], node[1]] = [
                    area.type.value,
                        area.start,
                        area.end,
                        area.pos,
                        area.pos,
                    ]
                elif area.axis == 1:
                    observation[:, node[0], node[1]] = [
                    area.type.value,
                        area.pos,
                        area.pos,
                        area.start,
                        area.end,
                    ]
            else:
                pass
        return observation
