
import gym
import datetime
import pandas
import logging
import os

from stable_baselines3 import PPO
from stable_baselines3 import A2C
from collections import namedtuple
from typing import Tuple
from typing import Union
from stable_baselines3.common.env_util import make_vec_env


class StockAgent:

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

        self.agent_model = None
        self.internal_net_arch = None
        self.env = None


def main():
    env = make_vec_env("CartPole-v1", n_envs=4)

    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=25000)
    model.save("ppo_cartpole")

    del model  # remove to demonstrate saving and loading

    model = PPO.load("ppo_cartpole")

    obs = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()