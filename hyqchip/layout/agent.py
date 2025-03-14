import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3 import PPO

from ..element.package import CellType
from .env import QuantumChipletEnv


class FeatureCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, package_size=(22, 11)):
        super(FeatureCNN, self).__init__(observation_space, features_dim=package_size[0]*package_size[1]*len(CellType))
        self.package_size = package_size
        
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

        with torch.no_grad():
            sample_input = torch.zeros(1, 1, *package_size)
            cnn_out_dim = self.cnn(sample_input).shape[1]
        self.fc = nn.Linear(cnn_out_dim, 256)

        self.fc = nn.Sequential(
            nn.Linear(cnn_out_dim, 512),
            nn.ReLU(),
            nn.Linear(512, package_size[0] * package_size[1] * len(CellType))
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
            "features_extractor_kwargs": {"package_size": self.package_size}
            }
        self.model = PPO("CnnPolicy", self.env, policy_kwargs=policy_kwargs, verbose=1)

        if model_path:
            self.model.load(model_path)
    
    def select_action(self, state):
        feature = self.model.policy(state)
        feature = feature.view(self.package_size[0], self.package_size[1], len(CellType))
        action_prob = torch.softmax(feature, dim=-1)
        action = torch.argmax(action_prob, dim=-1)
        return action.cpu().numpy()
    
    def train(self, total_timesteps=10000):
        print("Training agent...")
        self.model.learn(total_timesteps=total_timesteps)

    def save(self, path="ppo_quantum_chip_agent.zip"):
        self.model.save(path)
