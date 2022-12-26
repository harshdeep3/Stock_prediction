
import logging
import pandas_datareader as web

from datetime import datetime
from sklearn.preprocessing import StandardScaler
from stable_baselines3 import PPO
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
from stockEnv import StockEnv

START = datetime(2010, 7, 1)
END = datetime.today()
STOCK_NAME = 'AMZN'
SMA_TIME = 20
EMA_TIME = 20
RSI_TIME = 14
scaler = StandardScaler()


def get_data(stock_name, start_date, end_date):
    """[summary]
    This gets the data which is used by the agent.
    Returns:
        dataframe [dataframe]: This returns the dataframe which is used for the state and by the
        agent
    """
    df = web.DataReader(stock_name, 'yahoo', start_date, end_date)
    # reset the index of the data to normal ints so df['Date'] can be used
    df.reset_index()
    return df


class StockAgent:

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

        self.agent_model = None
        self.internal_net_arch = None
        self.env = None

    def set_env(self, env) -> None:
        """
        This function sets the environment in which the agent will learn and interact with. This can be a custom or
        pre-made environment.
        """
        self.env = env

    def create_model(self) -> None:
        """
        This create the model of the agent. The model will be created for the training and demonstration of the agent.

        :param hyp_lr: The learning rate
        :param hyp_n_steps: The number of steps to run for each environment per update
        :param hyp_batch_size: Minibatch size
        :param hyp_n_epoch: Number of epoch when optimizing the surrogate loss
        :param hyp_gamma: Discount factor
        :param hyp_gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        :param hyp_ent_coef: Entropy coefficient for the loss calculation
        :param att_tensorboard_logs: Directory location used to the debug info, can be used by tensorboard to view as
            graphs.
        """

        self.agent_model = PPO(policy=self.internal_net_arch, env=self.env)

    def train(self):

        self.create_model()

        # below 2500 train does not show up on tensorboard, anything below will remove train tag
        trial_timestep = 2500
        # timesteps is 25000 --- save the model every 2500 episodes
        for i in range(trial_timestep):
            self.agent_model.learn(total_timesteps=trial_timestep)

            self.save_model("saved_model/a2c")

    def set_policy(self) -> None:
        """
        This sets the policy that will be used by the model. This can set a custom policy or select one already created
        by stable baseline.
        """
        self.internal_net_arch = "MlpPolicy"


def main():
    data = get_data(STOCK_NAME, START, END)
    agent = StockAgent
    agent.set_policy()

    env = StockEnv(data)
    agent.set_env(env)

    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=1)
    model.save("ppo_cartpole")

    obs = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, episode_done, done, info = env.step(action)
        print(f"obs -> {obs}\nreward -> {rewards}\ndone -> {done}")
        # env.render()