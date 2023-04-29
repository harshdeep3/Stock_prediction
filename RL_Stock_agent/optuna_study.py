
import datetime
import optuna
import logging
import stockEnv
from stable_baselines3 import DQN
import os
import MT5_Link as link
import MetaTrader5 as mt5

from collections import namedtuple
from stable_baselines3.common.evaluation import evaluate_policy
from optuna import trial

LOGGING_LEVEL = logging.DEBUG
NUM_TRAILS = 10

# Hyper parameter options, to reduce the search space, specific values have been chosen.
GAMMA_CHOICES = [0.99]
LEARNING_RATE_CHOICES = [3e-4]
BATCH_SIZE_CHOICES = [64]
GAE_LAMBDA_CHOICES = [0.95]
ENT_COEF_CHOICES = [0.0]
N_STEPS_CHOICES = [2, 200, 2048, 10000, 20000, 40000]
N_EPOCH_CHOICES = [10]

hyperParameters = namedtuple('hyperParameters', ['lr', 'n_steps', 'batch_size', 'n_epoch', 'gamma',
                                                 'gae_lambda', 'ent_coef'])


def sample_hyperparameters(optuna_trial) -> hyperParameters:
    """
    This function suggest a value for the hyperparameters from a set list.
    :param optuna_trial: The trail which is suggesting the sample
    :return: the namedtuple storing the chosen hyperparameters
    """
    gamma = optuna_trial.suggest_categorical('gamma', GAMMA_CHOICES)
    lr = optuna_trial.suggest_categorical('lr', LEARNING_RATE_CHOICES)
    batch_size = optuna_trial.suggest_categorical('batch_size', BATCH_SIZE_CHOICES)
    gae_lambda = optuna_trial.suggest_categorical('gae-lambda', GAE_LAMBDA_CHOICES)
    ent_coef = optuna_trial.suggest_categorical('ent_coeff', ENT_COEF_CHOICES)
    n_steps = optuna_trial.suggest_categorical('n_steps', N_STEPS_CHOICES)
    n_epoch = optuna_trial.suggest_categorical('n_epoch', N_EPOCH_CHOICES)

    return hyperParameters(lr=lr, n_steps=n_steps, n_epoch=n_epoch, batch_size=batch_size, gamma=gamma,
                           gae_lambda=gae_lambda, ent_coef=ent_coef)


def run_trials(agent, env, optuna_trial: trial, trial_name: str) -> float:
    """
    This function runs the study and store the results in a csv file. This study find the best combination from the
    list.

    :param agent: the agent controller
    :param optuna_trial: the study trail
    :param trial_name: where the best hyperparameters are stored.
    :return: the mean reward from the study
    """
    now = datetime.datetime.now().strftime('%Y-%m-%d_%H%M%S')
    with open(trial_name, 'a', newline='') as file:
        file.write(f"{now},")
    env.reset()
    info = "saved_Files/log_dir"
    # sample from the choices available as a starting point
    # sampled_parameters = sample_hyperparameters(optuna_trial)

    agent.learn(total_timesteps=10000000)
    agent.save("saved_Files/saved_model/ppo")

    mean_reward, std_reward = evaluate_policy(agent, env, n_eval_episodes=2)

    if os.path.exists(trial_name):
        with open(trial_name, 'a', newline='') as file:
            # save the mean after the study is complete
            file.write(f"{mean_reward}")
            file.write('\n')

    return mean_reward


def main() -> None:
    """
    This creates the agent controller and allows the optuna study to occur. This will get optimised hyperparameter.
    """

    # logging
    now = datetime.datetime.now().strftime('%Y-%m-%d_%H%M%S')
    # logging.basicConfig(filename=f'controller_log/{now}_optuna_study_DQN.log',
    #                     level=LOGGING_LEVEL,
    #                     format='%(asctime)s :: %(levelname)s :: %(module)s :: %(processName)s'
    #                            ' :: %(funcName)s :: line #-%(lineno)d :: %(message)s')

    # file where the study results will be saved
    trial_name = f'saved_Files/{now}_optuna_study_DQN.csv'
    mt5_obj = link.MT5Class()
    mt5_obj.login_to_metatrader()
    mt5_obj.get_acc_info()

    start = datetime.datetime(2010, 7, 1).strftime("%Y-%m-%d")
    end = datetime.datetime.now().strftime("%Y-%m-%d")

    timeframe = mt5.TIMEFRAME_D1
    symbol = 'USDJPY'
    count = 8500  # get 8500 data points

    data = link.get_historic_data(fx_symbol=symbol, fx_timeframe=timeframe, fx_count=count)
    if data is None:
        print("Error: Data not recieved!")
    else:
        env = stockEnv.StockMarketEnv(data)
        agent = DQN('MlpPolicy', env, verbose=1)

        study = optuna.create_study(direction='maximize')

        # optuna optimisation
        study.optimize(lambda trail: run_trials(agent, env, trail, trial_name), n_trials=NUM_TRAILS)


if __name__ == '__main__':
    main()
