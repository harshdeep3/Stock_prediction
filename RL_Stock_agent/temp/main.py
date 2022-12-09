import gym
import pandas_datareader as web
from stable_baselines3 import PPO

from Env import StockEnv
from datetime import datetime

START = datetime(2010, 7, 1)
END = datetime.today()
STOCKNAME = 'BTC-USD'


def get_data(stock_name, start, end):
    df = web.DataReader(stock_name, 'yahoo', start, end)
    # reset the index of the data to normal ints so df['Date'] can be used
    df.reset_index()
    return df


def main():
    rl_env = gym.make('CartPole-v1')
    model = PPO('MlpPolicy', rl_env, verbose=1)

    model.learn(total_timesteps=10000)

    obs = rl_env.reset()

    for _ in range(1000):
        action, _state = model.predict(obs)
        obs, reward, done, info = rl_env.step(action)
        rl_env.render()
        if done:
            obs = rl_env.reset()


if __name__ == "__main__":
    data = get_data(STOCKNAME, START, END)
    env = StockEnv(data)

    print(env.action_space)

