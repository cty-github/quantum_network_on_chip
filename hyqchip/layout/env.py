import numpy as np
import gymnasium as gym
from gymnasium import spaces

from ..element.package import CellType, FabricateModel, Package


class QuantumChipletEnv(gym.Env):
    def __init__(self, package_size=(22, 11), fabricate_model=FabricateModel(), template='string', patience=100):
        super(QuantumChipletEnv, self).__init__()
        self.package_size = package_size
        self.fabricate_model = fabricate_model
        self.template = template

        self.package: Package = None
        self.state = np.zeros(self.package_size, dtype=np.int8)  # 0: Empty, 1: Compute, 2: Transfer
        self.score = 0

        self.action_space = spaces.Box(low=0, high=len(CellType)-1, shape=self.package_size, dtype=np.int8)
        self.observation_space = spaces.Box(low=0, high=len(CellType)-1, shape=(1, *package_size), dtype=np.int8)

        self.patience = patience
        self.best_reward = -np.inf
        self.no_improve_steps = 0

    def reset(self, seed=None, options=None):
        print('Resetting environment...')
        self.package = Package(self.package_size, self.fabricate_model, template=self.template)
        self.state = self.package.type_matrix
        self.best_reward = -np.inf
        self.no_improve_steps = 0
        return np.expand_dims(self.state, axis=0), {}

    def step(self, action):
        print('Taking action:')
        action = np.clip(np.round(action), 0, len(CellType)-1).astype(np.int8)
        print(action)
        for i in range(self.package_size[0]):
            for j in range(self.package_size[1]):
                if action[i, j]:
                    self.state[i, j] = action[i, j]
        self.package.update_all_cells(self.state)
        self.package.determine_die_and_coupler()

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
    
    def render(self, mode='human'):
        self.package.draw()

class StripQuChipEnv(QuantumChipletEnv):
    def __init__(self, package_size=(22, 11), fabricate_model=FabricateModel(), template='string', patience=100):
        super(StripQuChipEnv, self).__init__(package_size, fabricate_model, template, patience)

class RadialQuChipEnv(QuantumChipletEnv):
    def __init__(self, package_size=(22, 11), fabricate_model=FabricateModel(), template='string', patience=100):
        super(RadialQuChipEnv, self).__init__(package_size, fabricate_model, template, patience)    

class GridQuChipEnv(QuantumChipletEnv):
    def __init__(self, package_size=(22, 11), fabricate_model=FabricateModel(), template='string', patience=100):
        super(GridQuChipEnv, self).__init__(package_size, fabricate_model, template, patience)