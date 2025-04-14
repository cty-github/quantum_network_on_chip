import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from .env import QuantumChipletEnv


class FeatureCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, dim, package_size=(22, 11)):
        super(FeatureCNN, self).__init__(observation_space, features_dim=dim)
        self.package_size = package_size

        self.cnn = nn.Sequential(
            nn.Conv2d(5, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            sample_input = torch.zeros(1, 5, *package_size)
            cnn_out_dim = self.cnn(sample_input).shape[1]
        self.fc = nn.Linear(cnn_out_dim, 256)

        self.fc = nn.Sequential(
            nn.Linear(cnn_out_dim, 512),
            nn.ReLU(),
            nn.Linear(512, self.features_dim),
        )

    def forward(self, x):
        x = self.cnn(x)
        x = self.fc(x)
        return x


class LayoutAgent:
    def __init__(self, env: QuantumChipletEnv, model_path=None):
        self.env = env
        self.package_size = self.env.package_size

        policy_kwargs = {
            "features_extractor_class": FeatureCNN,
            "features_extractor_kwargs": {
                "package_size": self.package_size,
                "dim": sum(env.action_space.nvec),
            },
        }
        self.model = PPO("CnnPolicy", self.env, policy_kwargs=policy_kwargs, verbose=1)

        if model_path:
            self.model.load(model_path)

    def select_action(self, state):
        state = torch.as_tensor(state, dtype=torch.int32).unsqueeze(0)
        logits = self.model.policy(state)
        # 每一块特征做 softmax
        with torch.no_grad():
            # 提取特征并获取动作 logits
            features = self.model.policy.features_extractor(state)
            logits = self.model.policy.action_net(features)
        split_logits = torch.split(logits, list(self.env.action_space.nvec), dim=1)
        actions = [torch.argmax(logits_dim, dim=1).item() for logits_dim in split_logits]

        return np.array(actions)

    def train(self, total_timesteps=10000):
        print("Training agent...")
        self.model.learn(total_timesteps=total_timesteps)

    def save(self, path="ppo_quantum_chip_agent.zip"):
        self.model.save(path)

    def load(self, path="ppo_quantum_chip_agent.zip"):
        self.model.load(path)
