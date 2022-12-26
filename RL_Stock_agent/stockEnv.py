import gym
from typing import Tuple


class StockEnv(gym.Env):

    def __init__(self):
        super(StockEnv, self).__init__()

        self.action_space = None
        self.observation_space = None

    def get_cur_obs(self) -> gym.spaces.MultiDiscrete:
        pass

    def step(self, action) -> Tuple[gym.spaces.MultiDiscrete, float, bool, dict]:

        return obs, reward, done, info

    def reset(self) -> gym.spaces.MultiDiscrete:

        return observation  # reward, done, info can't be included

    def render(self, mode="human") -> None:
        pass

    def close (self) -> None:
        pass