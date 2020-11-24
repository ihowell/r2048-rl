import numpy as np

import r2048

import gym
from gym import spaces


class R2048Env(gym.Env):
    def __init__(self):
        self.observation_space = spaces.MultiDiscrete([18 for _ in range(16)])
        self.action_space = spaces.Discrete(4)
        self._state = r2048.new()

    def reset(self):
        self._state = r2048.new()
        return self._state

    def step(self, action):
        state, reward, done = r2048.step(self._state, action)
        self._state = state
        return np.array(state), np.array(reward), done, {}

    def render(self, mode='human'):
        return None
