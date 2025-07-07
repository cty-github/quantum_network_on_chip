import os
import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy

from .env import QuNoCEnv
from ..element.package import Package


class LayoutFeaturesExtractor(BaseFeaturesExtractor):
    """
    A CNN feature extractor for the QuNoCEnv environment.

    Attributes:
        cnn (nn.Sequential): The CNN model.
        linear (nn.Sequential): The linear layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        """
        Initialize the CNN feature extractor.

        Args:
            observation_space (spaces.Box): The observation space.
            features_dim (int): The dimension of the output features.
        """
        super().__init__(observation_space, features_dim)
        in_channels = observation_space.shape[0]  # Should be 5

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

        with torch.no_grad():
            dummy_input = torch.zeros((1, *observation_space.shape))
            n_flatten = self.cnn(dummy_input).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations.float()))


class LayoutPolicy(ActorCriticPolicy):
    """
    A custom policy for the QuNoCEnv environment.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            features_extractor_class=LayoutFeaturesExtractor,
            features_extractor_kwargs=dict(features_dim=256),
            **kwargs,
        )


class LayoutAgent:
    """
    A reinforcement learning agent for the QuNoCEnv environment.

    Attributes:
        env (QuNoCEnv): The environment to train the agent.
        model (PPO): The PPO model.
        result_dir (str): Directory to save the results.
    """

    def __init__(self, env: QuNoCEnv, pre_trained_model_path: str = None, result_dir: str = None):
        """
        Initialize the agent.

        Args:
            env (QuNoCEnv): The environment to train the agent.
            pre_trained_model_path (str): Path to the pre-trained model.
            result_dir (str): Directory to save the results.
        """
        self.env = env
        self.model = PPO(LayoutPolicy, env, verbose = 1)
        if pre_trained_model_path:
            self.load_model(pre_trained_model_path)
        self.result_dir = result_dir

    def train(self, total_timesteps: int = 10000):
        """
        Train the agent.

        Args:
            total_timesteps (int): Total timesteps for training.
        """
        print("Training agent...")
        self.model.learn(total_timesteps = total_timesteps)
        self.save_model()
    
    def design(self) -> Package:
        """
        Design the layout using the trained agent.

        Returns:
            Package: The designed package.
        """
        print("Designing layout...")
        state, _ = self.env.reset()
        print("Initial score:", self.env.estimate())
        done = False
        while not done:
            action, _ = self.model.predict(state, deterministic=True)
            state, _, done, _, _ = self.env.step(action, record=True)
            print("New action:", action)
            print("New score:", self.env.estimate())
        return self.env.best_package

    def load_model(self, filepath: str = None):
        """
        Load a pre-trained PPO model from a file.

        Args:
            filepath (str): Path to the pre-trained model file. If None, uses the default path.
        """
        self.model.load(filepath if filepath else os.path.join(self.result_dir, "layout_agent.pth"))

    def save_model(self, filepath: str = None):
        """
        Save the PPO model to a file.

        Args:
            filepath (str): Path to save the model file. If None, uses the default path.
        """
        self.model.save(filepath if filepath else os.path.join(self.result_dir, "layout_agent.pth"))
