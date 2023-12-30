import numpy as np

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback

import datetime
import os
import logging
from stock_Env import Env
import MT5_Link as link
import MetaTrader5 as mt5
from stockEnv import StockMarketEnv as GenEnv
from stable_baselines3.common.env_util import make_vec_env


LOGGING_LEVEL = logging.DEBUG

class CustomCallback(BaseCallback):
    
    def __init__(self, verbose: int = 0):
        super(CustomCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        
        cash_left = np.mean(self.training_env.get_attr("cash_in_hand"))
        reward = np.mean(self.training_env.get_attr("reward"))
        total_net = np.mean(self.training_env.get_attr("total_net"))

        self.logger.record("cash_in_hand - ", cash_left)
        self.logger.record("reward - ", reward)
        self.logger.record("Total Net - ", total_net)
        
        return True

def main() -> None:
    """
    This creates the agent controller and allows the optuna study to occur. This will get optimised hyperparameter.
    """

    # logging
    now = datetime.datetime.now().strftime('%Y-%m-%d_%H%M%S')
    logging.basicConfig(filename=f'RL_Stock_agent/saved_Files/log_files/{now}_RPPO_log.log',
                        level=LOGGING_LEVEL,
                        format='%(asctime)s :: %(levelname)s :: %(module)s :: %(processName)s'
                               ' :: %(funcName)s :: line #-%(lineno)d :: %(message)s')

    # file where the study results will be saved
    trial_name = f'saved_Files/{now}_optuna_study_PPO_4_envs.csv'
    mt5_obj = link.MT5Class()
    mt5_obj.login_to_metatrader()
    mt5_obj.get_acc_info()
    start = datetime.datetime(2010, 7, 1).strftime("%Y-%m-%d")
    end = datetime.datetime.now().strftime("%Y-%m-%d")

    timeframe = mt5.TIMEFRAME_D1
    symbol = 'USDJPY'
    count = 13500  # get 8500 data points

    # env = make_vec_env(Env, n_envs=4)
    data = link.get_historic_data(fx_symbol=symbol, fx_timeframe=timeframe, fx_count=count)
    if data is None:
        print("Error: Data not recieved!")
        env = "CartPole-v1"
    else:
        # env = GenEnv(data)
        env = make_vec_env(Env, n_envs=1)

    if os.path.exists("RL_Stock_agent/saved_Files/saved_model/Rec_PPO"):
        agent = RecurrentPPO.load("RL_Stock_agent/saved_Files/saved_model/Rec_PPO")
    else:
        agent = RecurrentPPO('MlpLstmPolicy', env, verbose=1, tensorboard_log="RL_Stock_agent/saved_Files/logs")
    # running a recurrentPPO agent
    agent = RecurrentPPO('MlpLstmPolicy', env, verbose=1, tensorboard_log="RL_Stock_agent/saved_Files/logs")
    
    # train the agent 
    agent.learn(total_timesteps=10, callback=CustomCallback())
    
    # save the trained agent
    agent.save("RL_Stock_agent/saved_Files/saved_model/Rec_PPO")

    # load the saved agent
    agent = RecurrentPPO.load("RL_Stock_agent/saved_Files/saved_model/Rec_PPO")
    
    # reset env
    obs = env.reset()
    
    # evaluate 
    for _ in range(15):
        action, _ = agent.predict(obs)
        obs, rewards, _, _ = env.step(action)
        
    

if __name__ == '__main__':
    main()
    
    #logging
    now = datetime.datetime.now().strftime('%Y-%m-%d_%H%M%S')
    logging.basicConfig(filename=f'saved_Files/log_files/{now}.log',
                        level=LOGGING_LEVEL,
                        format='%(asctime)s :: %(levelname)s :: %(module)s :: %(processName)s'
                               ' :: %(funcName)s :: line #%(lineno)d :: %(message)s')


