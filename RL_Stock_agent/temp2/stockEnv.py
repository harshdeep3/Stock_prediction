from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
import gym
import yfinance as yf
import pandas as pd
import numpy as np

import yfinance as yf
import gym
from gym import spaces
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import A2C

class StockTradingEnv(gym.Env):
    """A stock trading environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(StockTradingEnv, self).__init__()

        # Define action and observation space
        self.action_space = spaces.Discrete(3)  # sell, hold, or buy
        self.observation_space = spaces.Box(low=0, high=1, shape=(6,))

        # Load data from Yahoo Finance
        self.df = yf.download("AAPL", start="2021-01-01", end="2021-12-31")
        self.df.reset_index(inplace=True)

        # Define state variables
        self.stock_owned = None
        self.cash_in_hand = None
        self.state = None

        # Set initial state
        self.reset()

    def reset(self):
        # Reset state variables
        self.stock_owned = 0
        self.cash_in_hand = 1000

        # Reset state
        self.state = [
            self.df.loc[0, 'Open'],
            self.df.loc[0, 'High'],
            self.df.loc[0, 'Low'],
            self.df.loc[0, 'Close'],
            self.df.loc[0, 'Adj Close'],
            self.cash_in_hand / (self.stock_owned + 1e-10)
        ]

        return np.array(self.state)

    def step(self, action):
        # Get current price
        current_price = self.df.loc[self.current_step, 'Close']

        # Sell
        if action == 0:
            self.stock_owned = 0
            self.cash_in_hand += self.stock_owned * current_price

        # Hold
        elif action == 1:
            pass

        # Buy
        elif action == 2:
            max_buy = int(self.cash_in_hand / current_price)
            shares_bought = max_buy
            self.stock_owned += shares_bought
            self.cash_in_hand -= shares_bought * current_price

        # Update state
        self.state = [
            self.df.loc[self.current_step, 'Open'],
            self.df.loc[self.current_step, 'High'],
            self.df.loc[self.current_step, 'Low'],
            self.df.loc[self.current_step, 'Close'],
            self.df.loc[self.current_step, 'Adj Close'],
            self.cash_in_hand / (self.stock_owned + 1e-10)
        ]

        # Update current step
        self.current_step += 1

        # Calculate reward
        reward = self.cash_in_hand + self.stock_owned * current_price - 1000

        # Check if done
        if self.current_step == len(self.df) - 1:
            done = True
        else:
            done = False

        return np.array(self.state), reward, done, {}

    def render(self, mode='human'):
        print(f'Step: {self.current_step}')
        print(f'Stock Owned: {self.stock_owned}')
        print(f'Cash in Hand: {self.cash_in_hand}')


# Create agent and train
# model = A2C("MlpPolicy", env, verbose=1)

env = StockTradingEnv()
check_env(env)

model = DQN('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=10000)
