import LinearModel as lm
import numpy as np
import pandas as pd
import pandas_datareader as web
import gym
from gym import spaces

from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env
from datetime import datetime
import itertools
import pickle
from sklearn.preprocessing import StandardScaler

start = datetime(2010, 7, 1)
end = datetime.today()
stockName = 'AMZN'
sma_time = 20
ema_time = 20
rsi_time = 14
scaler = StandardScaler()


def get_data(stock_name, data_start, data_end):
    """[summary]
    This gets the data which is used by the agent.
    Returns:
        dataframe [dataframe]: This returns the dataframe which is used for
                                                    the state and by the agent
    """
    df = web.DataReader(stock_name, 'yahoo', data_start, data_end)
    # reset the index of the data to normal ints so df['Date'] can be used
    df.reset_index()
    return df


def get_scaler(env):
    """[summary]
    This scales the states to fit the standard scaler
    Args:
        env ([type]): [description] This is the stock market environment
    Returns:
         scaler	[type]: [description] returns a sklearn object scaling the states.
    """
    states = []
    for _ in range(env.n_step):
        # getting a random action
        action = np.random.choice(env.action_space)
        # getting the state after the action if taken
        state, _, _, done, _ = env.step(action)
        states.append(state)
        if done:
            # if the data runs out the env resets
            env.reset()
            break
    # fitting the scaller according to the states
    scaler.fit(states)
    return scaler


def calculate_rsi(data, time=rsi_time):
    """[summary]
    RSI is an indicator which is used by traders, to look at the strength of
    the current price movement. This returns the RSI value for each state

    Args:
            data ([Dataframe]): The data used to calculate the RSI at a state
            time ([int], optional): RSI calculations use a time frame which it
                                                        spans to get the value. Defaults to rsi_time, which is set to 14.

    Returns:
            [float]: The RSI value for that movement.
    """

    global rsi_time
    try:
        diff = data.Close.diff()
        # this preservers dimensions off diff values
        up = 0 * diff
        down = 0 * diff
        # up change is equal to the positive difference
        up[diff > 0] = diff[diff > 0]
        # down change is equal to negative deifference,
        down[diff < 0] = diff[diff < 0]

        avgerageUp = up.ewm(span=time, min_periods=time).mean()
        avgerageDown = down.ewm(span=time, min_periods=time).mean()

        rs = abs(avgerageUp / avgerageDown)
        # Change the value of the rsi used to the value displayed on the app
        rsi_time = time
        return (100 - 100 / (1 + rs))

    except Exception as e:
        print("Failed! Error", e)


def calculate_sma(data, time=sma_time):
    """[summary]
    This calculates the values for the simple moving average.
    Args:
            data ([dataframe]): Data used to calculate the sma
            time (int, optional): This is the time period the moving avergae being
            calculated. Defaults to sma_time, which is set to 20 intially.

    Returns:
            [type]: Moving average values
    """
    global sma_time
    # Change the value of the sma used to the value displayed on the app
    sma_time = time
    return data['Close'].rolling(time).mean()


def calculate_ema(data, time=ema_time):
    """[summary]
    This calculates the exponential moving average, this gives more importance
    to the newer data.
    Args:
            data ([dataframe]): Data used to calculate the ema
            time (int, optional): This is the time period the moving avergae being
            calculated. Defaults to ema_time, which is set to 20 intially.

    Returns:
            [type]: [description]
    """
    global ema_time
    # Change the value of the ema used to the value displayed on the app
    ema_time = time
    return data['Close'].ewm(span=time, min_periods=0, adjust=False).mean()


class Env(gym.Env):
    def __init__(self, data, initial_investment=20000):
        # getting the data and using it to get the number of day in the history and the number of stock
        self.stock_price_history = data
        # only working with one stock at a time
        self.n_stock = 1
        # the number of days in the data
        self.n_step = self.stock_price_history.shape[0]

        self.state_size_used = 11
        # instance attributes
        self.initial_investment = initial_investment
        self.cur_step = None  # -> the current day in the history
        self.stock_owned = None  # -> stocks owned as a vector
        self.stock_open = None  # -> stock open price for the day as a vector
        self.stock_high = None  # -> stock highest price for the day as a vector
        self.stock_low = None  # -> stock lowest price for the day as a vector
        self.stock_close = None  # -> stock close price for the day as a vector
        # Adjusted close is the closing price after adjustments for all applicable
        # splits and dividend distributions.
        self.stock_AdjClose = None  # -> stock adj close price for the day as a vector
        self.stock_Volume = None  # -> stock volume traded for the day as a vector
        self.rsi = None  # -> indicator used in trading, to tell if the market is
        # over bought or sold
        self.sma = None  # -> indicator to average the last x days
        # -> indicator to average the last x days but the newer have more importance.
        self.ema = None

        # obs_space = spaces.Dict({
        #     'stock_owned': spaces.Box(low=0, high=np.inf, shape=(self.n_stock,), dtype=np.float32),
        #     'stock_price': spaces.Box(low=0, high=np.inf, shape=(9,), dtype=np.float32),
        #     'cash_in_hand': spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32)
        # })

        self.state_dim = self.n_stock * 10 + 1

        self.cash_in_hand = None  # -> cash left after invesment
        # possibilities of the action -> 3 actions (buy, sell, hold)
        # possibilities = 3^number of stocks
        self.action_space = spaces.Discrete(3 ** self.n_stock)

        # itertools implement a number of iterator building blocks
        # itertools.product computes the cartesian product of input iterables,
        # equivlant to nested forloops
        self.action_list = list(
            map(list, itertools.product([0, 1, 2], repeat=self.n_stock)))

        # to start the data for day 0 and get all the info for day 0
        self.reset()

    def reset(self):
        """[summary]
        This reset the enviroment, starting from first data point in the data.

        Returns:
                [type]: Return an observation of the the environment
        """
        self.cur_step = 0
        self.stock_owned = np.zeros(self.n_stock)

        # info from Yahoo finance
        self.stock_open = self.stock_price_history.Open[self.cur_step]
        self.stock_high = self.stock_price_history.High[self.cur_step]
        self.stock_low = self.stock_price_history.Low[self.cur_step]
        self.stock_close = self.stock_price_history.Close[self.cur_step]
        self.stock_AdjClose = self.stock_price_history['Adj Close'][self.cur_step]
        self.stock_Volume = self.stock_price_history.Volume[self.cur_step]
        # indicators
        # rsi ema and sma has nan for the first couple of values.
        rsi = calculate_rsi(self.stock_price_history)
        if np.isnan(rsi[self.cur_step]):
            self.rsi = 0
        else:
            self.rsi = rsi[self.cur_step]

        sma = calculate_sma(self.stock_price_history)
        ema = calculate_ema(self.stock_price_history)
        # as this use previous data to get the current value, this can be nan
        # at the first few time steps, this is to remove them
        if np.isnan(sma[self.cur_step]):
            self.sma = self.stock_close
        else:
            self.sma = sma[self.cur_step]

        if np.isnan(ema[self.cur_step]):
            self.ema = self.stock_close
        else:
            self.ema = ema[self.cur_step]

        self.cash_in_hand = self.initial_investment
        return self._get_obs()

    def step(self, action):
        """[summary]
        This takes an action and updates state. It also store the reward
        for each episode
        Args:
            action ([type]): the action to be take, can buy, sell, or hold
        Returns:
            [type]: This returns the observation of state, the reward for that state,
                    boolean to show if the episode has ended, boolean to state if it is at
                    the of the data and the	current protofolio.
        """
        # day reward
        r = 0
        # today reward
        cur_val = 0
        # data end
        done = False
        # episode end
        episode_done = False
        # current value
        prev_val = self._get_val()

        # info from Yahoo finance
        self.stock_open = self.stock_price_history.Open[self.cur_step]
        self.stock_high = self.stock_price_history.High[self.cur_step]
        self.stock_low = self.stock_price_history.Low[self.cur_step]
        self.stock_close = self.stock_price_history.Close[self.cur_step]
        self.stock_AdjClose = self.stock_price_history['Adj Close'][self.cur_step]
        self.stock_Volume = self.stock_price_history.Volume[self.cur_step]
        # indicators
        # rsi ema and sma has nan for the first couple of values.
        rsi = calculate_rsi(self.stock_price_history)
        if np.isnan(rsi[self.cur_step]):
            self.rsi = 0
        else:
            self.rsi = rsi[self.cur_step]

        sma = calculate_sma(self.stock_price_history)
        ema = calculate_ema(self.stock_price_history)

        if np.isnan(sma[self.cur_step]):
            self.sma = self.stock_close
        else:
            self.sma = sma[self.cur_step]

        if np.isnan(ema[self.cur_step]):
            self.ema = self.stock_close
        else:
            self.ema = ema[self.cur_step]

        # doing the action
        self._trade(action)

        # value after the action is taken
        cur_val = self._get_val()
        # reward is the difference between the previous val and the cur value
        r = cur_val - prev_val
        self.cur_step += 1
        if self.cur_step % 30 == 0:
            episode_done = True
        # if the last day of the data then done is true
        done = self.cur_step == self.n_step - 1
        info = {'cur_val': cur_val}
        return self._get_obs(), r, episode_done, done, info

    def _get_obs(self):
        obs = {
            'stock_owned': self.stock_owned,
            'stock_price': np.array([
                self.stock_open,
                self.stock_high,
                self.stock_low,
                self.stock_close,
                self.stock_AdjClose,
                self.stock_Volume,
                self.rsi,
                self.sma,
                self.ema
            ]),
            'cash_in_hand': self.cash_in_hand
        }
        return obs

    def _get_val(self):
        # protfolio
        """[summary]
        This gets the protofolio, the total money left after all of the stock are sold.
        Worked out by using #of stock owned * price pre stock + cash left
        Returns:
                [type]: protofolio, total after everything is sold
        """

        return self.stock_owned * self.stock_close + self.cash_in_hand

    def _trade(self, action):
        """[summary]
        This simulate the buying/selling of a stock. This is done by incrementing over each day
        buying if there is enough cash left.
        Args:
                action ([type]): either buy, sell or holding represented as a number
        """
        # index the action we want to perform
        # 0 = sell
        # 1 = hold
        # 2 = buy
        action_vec = self.action_list[action]
        allactions = []
        sell_index = []
        buy_index = []

        for i, a in enumerate(action_vec):
            if a == 0:
                # sell stocks
                sell_index.append(i)
            elif a == 2:
                # buy stocks
                buy_index.append(i)

        if sell_index:
            for _ in sell_index:
                self.cash_in_hand += self.stock_close * self.stock_owned
                self.stock_owned = 0
        elif buy_index:
            can_buy = True
            while can_buy:
                for i in buy_index:
                    # buy a single stock on loop
                    # if there are multiple stocks then the loop iterates over each stock and buys 1 stock
                    # (if it can) for the stocks avaible each iteration.
                    if self.cash_in_hand > self.stock_close:
                        self.stock_owned += 1  # buy one share
                        self.cash_in_hand -= self.stock_close
                    else:
                        # if the cash left is lower the stock price then stop buying
                        can_buy = False

        return allactions


class Agent(object):
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        # discount rate
        self.gamma = 0.95
        # exploration rate
        self.epsilon = 0.9
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9975
        self.model = DQN("MlpPolicy", make_vec_env(lambda: CustomEnv(state_size, action_size)), verbose=0)

    def act(self, state):
        """Get the actions which will be chosen.
        This can be a random action (explore) or the action which gets the best rewards (exploit).
        """
        if np.random.rand() <= self.epsilon:
            # exploring
            return np.random.choice(self.action_size)
        # exploiting
        act_values = self.model.predict(np.array([state]))[0]
        return np.argmax(act_values)

    def train(self, state, action, reward, next_state, done):
        """Train the model and update the exploration rate."""
        self.model.learn(total_timesteps=1, reset_num_timesteps=False)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        """Load the previously saved weight."""
        self.model = DQN.load(name)

    def save(self, name):
        """Save the weights."""
        self.model.save(name)


def play_one_episode(agent, env):
    """[summary]
    Goes through the data, while and improving it knowledge
    Args:
            agent ([type]): agent which learn
            env ([type]): environmnet where the agent is used
    Returns:
            [type]: the cur protfolio, episodic rewards, dayrewards.
    """
    # note: after transforming states are already 1xD
    state = env.reset()
    state = scaler.transform([state])
    # to store the rewards for day
    dayReward = []
    # to store the rewards for episode
    allRewards = []
    # end of the data
    done = False
    episodeReward = []
    while not done:
        episodeDone = False
        if not episodeDone and not done:
            # getting the action
            action = agent.act(state)
            # going through a step to get s', r, epsiode and data end, and protofolio
            next_state, reward, episodeDone, done, info = env.step(action)
            episodeReward.append(reward[0])
            # storing the day rewards
            dayReward.append(reward[0])
            next_state = scaler.transform([next_state])
            # learning using SGD
            agent.train(state, action, reward, next_state, episodeDone)
            # change in state
            state = next_state
            # if episode is complete
            if episodeDone or done:
                # if the episode over then get the sum of the rewards
                # for the episodix rewards
                allRewards.append(sum(episodeReward))
                episodeReward = []
    return info['cur_val'], allRewards, dayReward


def main(data, dataType, train_no=1):
    """[summary]
    This is the main funciton, this runs the training and testing of the agent
    Args:
            data ([type]): Stock data which will be used
            dataType ([type]): test or train
            train_no (int, optional): The number of times the agent is trained
                                                            choosen by the user. Defaults to 1.
    Returns:
            [type]: for training it returns the losses for the training, the cur
                         protfolio,dayrewards and final val. For testing it returns
                         the profit from the test, protfolio, dayrewards and final val.
    """
    # num_episode = 20
    initial_investment = 20000
    numOfTrain = 2100
    # 2100 train values, 550 test values

    traindata = data[:numOfTrain]
    testdata = data[numOfTrain:]
    # so the index starts at 0
    testdata.reset_index(inplace=True)

    if dataType == "train":
        # training
        env = Env(traindata, initial_investment)
        stateSize = env.state_dim
        action_size = len(env.action_space)
        agent = Agent(stateSize, action_size)
        scaler = get_scaler(env)

        protfolio = []
        actions_taken = []
        training_val = []
        # run the trading agent
        for i in range(1, train_no + 1):
            if train_no < 20:
                print(i)
            elif train_no < 50:
                if train_no % 5 == 0:
                    print(i)
            elif train_no < 200:
                if train_no % 10 == 0:
                    print(i)
            elif train_no < 500:
                if train_no % 50 == 0:
                    print(i)
            elif train_no > 500:
                if train_no % 100 == 0:
                    print(i)
            val, protfolio, dayRewards = play_one_episode(agent, env)
            training_val.append(val[0])
        agent.save('linear.pt')
        with open('scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)

        return agent.model.losses, protfolio, dayRewards, training_val

    elif dataType == "test":
        env = Env(testdata, initial_investment)
        stateSize = env.state_dim
        action_size = len(env.action_space)
        agent = Agent(stateSize, action_size)
        with open(f'scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)

        get_scaler(env)
        agent.epsilon = 0.01
        agent.load('linear.pt')

        val, protfolio, dayRewards = play_one_episode(agent, env)

        profit = []
        money = env.initial_investment
        for i in range(len(protfolio)):
            money += protfolio[i]
            profit.append(money)

        return profit, protfolio, dayRewards, val


if __name__ == "__main__":
    data = get_data(stockName, start, end)

    # 	print(data.Open[0])
    losses, protfolio, day_rewards, training_val, count = main(data, "train")

    # 	print()
    print(main(data, "test"))
