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
    env = gym.make('CartPole-v1')
    model = PPO('MlpPolicy', env, verbose=1)

    model.learn(total_timesteps=10000)

    obs = env.reset()

    for _ in range(1000):
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            obs = env.reset()


if __name__ == "__main__":
    data = get_data(STOCKNAME, START, END)
    env = StockEnv(data)

    print(data.max())

