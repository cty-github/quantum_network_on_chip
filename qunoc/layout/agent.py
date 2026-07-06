import os

import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from .env import *


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


class RolloutProgressCallback(BaseCallback):
    """
    Print PPO rollout collection progress before each policy update.
    """

    def __init__(self):
        super().__init__()
        self.progress_interval = 8
        self.rollout_index = 0
        self.rollout_steps = 0
        self.rollout_target_steps = 0
        self.rollout_episodes = 0

    def _on_rollout_start(self) -> None:
        self.rollout_index += 1
        self.rollout_steps = 0
        self.rollout_episodes = 0
        num_envs = getattr(self.training_env, "num_envs", 1)
        self.rollout_target_steps = self.model.n_steps * num_envs
        print(
            f"[PPO collect] rollout={self.rollout_index} "
            f"start target={self.rollout_target_steps} env_steps",
            flush=True,
        )

    def _on_step(self) -> bool:
        dones = self.locals.get("dones")
        if dones is None:
            num_envs = getattr(self.training_env, "num_envs", 1)
        else:
            num_envs = len(dones)
            self.rollout_episodes += sum(bool(done) for done in dones)

        self.rollout_steps += num_envs
        should_print = (
            self.rollout_steps % self.progress_interval == 0
            or self.rollout_steps >= self.rollout_target_steps
        )
        if should_print:
            print(
                f"[PPO collect] rollout={self.rollout_index} "
                f"collected={self.rollout_steps}/{self.rollout_target_steps} "
                f"total_timesteps={self.num_timesteps} "
                f"episodes_done={self.rollout_episodes}",
                flush=True,
            )
        return True

    def _on_rollout_end(self) -> None:
        print(
            f"[PPO collect] rollout={self.rollout_index} complete; "
            "PPO update starts",
            flush=True,
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
        self.model = PPO(
            LayoutPolicy,
            env,
            verbose=1,
            n_steps=256
        )
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
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=RolloutProgressCallback(),
        )
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
        """
        self.model = PPO.load(
            filepath if filepath is not None else os.path.join(self.result_dir, "layout_agent.pth"),
            env=self.env,
        )

    def save_model(self, filepath: str = None):
        """
        Save the PPO model to a file.
        """
        self.model.save(filepath if filepath is not None else os.path.join(self.result_dir, "layout_agent.pth"))
