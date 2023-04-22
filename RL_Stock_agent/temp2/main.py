from linearModel import LinearModel as Lm
import numpy as np
import pandas_datareader as web

from datetime import datetime
import itertools
import pickle
from sklearn.preprocessing import StandardScaler

start = datetime(2010, 7, 1)
end = datetime.today()
stockName = 'AMZN'
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


def calculateRsi(data, time=RSI_TIME):
    """[summary]
    RSI is an indicator which is used by traders, to look at the strength of the current price
    movement. This returns the RSI value for each state

    Args:
        data ([Dataframe]): The data used to calculate the RSI at a state
        time ([int], optional): RSI calculations use a time frame which it spans to get the value.
            Defaults to rsi_time, which is set to 14.

    Returns:
        [float]: The RSI value for that movement.
    """

    global RSI_TIME
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
        return 100 - 100 / (1 + rs)

    except Exception as e:
        print("Failed! Error", e)


def calculateSMA(data, time=SMA_TIME):
    """[summary]
    This calculates the values for the simple moving average.
    Args:
        data ([dataframe]): Data used to calculate the sma
        time (int, optional): This is the time period the moving avergae being calculated. Defaults
            to sma_time, which is set to 20 intially.

    Returns:
        [type]: Moving average values
    """
    global SMA_TIME
    # Change the value of the sma used to the value displayed on the app
    sma_time = time
    return data['Close'].rolling(time).mean()


def calculateEMA(data, time=EMA_TIME):
    """[summary]
    This calculates the exponential moving average, this gives more importance
    to the newer data.
    Args:
        data ([dataframe]): Data used to calculate the
        time (int, optional): This is the time period the moving avergae being calculated.
            Defaults to ema_time, which is set to 20 intially.

    Returns:
        [type]: [description]
    """
    global EMA_TIME
    # Change the value of the ema used to the value displayed on the app
    ema_time = time
    return data['Close'].ewm(span=time, min_periods=0, adjust=False).mean()


class Env:
    def __init__(self, data, initial_investment=20000, state_size_used=11):
        # getting the data and using it to get the number of day in the history and the number of stock
        self.stock_price_history = data
        # only working with one stock at a time
        self.n_stock = 1
        # the number of days in the data
        self.n_step = self.stock_price_history.shape[0]

        self.state_size_used = state_size_used
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

        self.state_dim = self.n_stock * 10 + 1

        self.cash_in_hand = None  # -> cash left after invesment
        # possibilities of the action -> 3 actions (buy, sell, hold)
        # possibilities = 3^number of stocks
        self.action_space = np.arange(3 ** self.n_stock)

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
        rsi = calculateRsi(self.stock_price_history)
        if np.isnan(rsi[self.cur_step]):
            self.rsi = 0
        else:
            self.rsi = rsi[self.cur_step]

        sma = calculateSMA(self.stock_price_history)
        ema = calculateEMA(self.stock_price_history)
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
        # episode end
        episodeDone = False
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
        rsi = calculateRsi(self.stock_price_history)
        if np.isnan(rsi[self.cur_step]):
            self.rsi = 0
        else:
            self.rsi = rsi[self.cur_step]

        sma = calculateSMA(self.stock_price_history)
        ema = calculateEMA(self.stock_price_history)

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
            episodeDone = True
        # if the last day of the data then done is true
        done = self.cur_step == self.n_step - 1
        info = {'cur_val': cur_val}
        return self._get_obs(), r, episodeDone, done, info

    def _get_obs(self):
        """[summary]
        This is the observation of the state, vector describing the state, using the number of
        stocks owned, price of each stock, indicators and the cash left

        Returns:
            [vector]: [description] The state [number of stocks owned, Open price, high price,
            low price, close price, rsi, sma, ema, cash left]
        """
        obs = np.empty(self.state_dim)

        # [,open, high, low, close, Adj Close, Volume, rsi, sma, ema]
        obs[self.n_stock:2 * self.n_stock] = self.stock_open
        obs[2 * self.n_stock:3 * self.n_stock] = self.stock_high
        obs[3 * self.n_stock:4 * self.n_stock] = self.stock_low
        obs[4 * self.n_stock:5 * self.n_stock] = self.stock_close
        obs[5 * self.n_stock:6 * self.n_stock] = self.stock_AdjClose
        obs[6 * self.n_stock:7 * self.n_stock] = self.stock_Volume
        obs[7 * self.n_stock:8 * self.n_stock] = self.rsi
        obs[8 * self.n_stock:9 * self.n_stock] = self.sma
        obs[9 * self.n_stock:10 * self.n_stock] = self.ema

        # cash left
        obs[-1] = self.cash_in_hand
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

        for _, a in enumerate(action_vec):
            if a == 0:
                # sell stocks
                sell_index.append(_)
            elif a == 2:
                # buy stocks
                buy_index.append(_)

        if sell_index:
            for _ in sell_index:
                self.cash_in_hand += self.stock_close * self.stock_owned
                self.stock_owned = 0
        elif buy_index:
            can_buy = True
            while can_buy:
                for _ in buy_index:
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
        self.epsilon_min = 0.01  # -> smallest epsilon value
        # after a testing the epsilon value goes 0.9 to around 0.01, exactly
        # what i want. So the decay is set to 0.998
        self.epsilon_decay = 0.9975  # -> decay epslion by 99.9%
        self.model = Lm(state_size, action_size)

    def act(self, state):
        """[summary]
        This gets the actions which will be chosen. This can be a random action
        (explore) or the action which gets the best rewards (exploit).
        Args:
            state ([type]):The state describe the state stocks owned, OHLC prices,
                                            indicators, and cash_left
        Returns:
            [type]: The choosen action
        """
        if np.random.rand() <= self.epsilon:
            # exploring
            return np.random.choice(self.action_size)
        # exploiting
        act_values = self.model.predict(state).detach().numpy()
        return np.argmax(act_values)

    def train(self, state, action, reward, next_state, done):
        """[summary]
        This trains the model, gets the target value. The epsilon value decreased each time this is run
        Args:
            state ([vector]): The current state describe the environment using the state vector
            action ([type]): The action that will be taken.
            reward ([type]): The change between the current protfolio and the one after the aciton
                is taken
            next_state ([vector]): Observation of the environment after the action is taken
            done (boolean): If the data is on the last day
        """
        # get the prediction of the next state
        # this yhat = r + gamma * max (preds according to the actions)
        target = reward + self.gamma * np.amax(self.model.predict(next_state).detach().numpy())
        # get the predict of the y
        target_full = self.model.predict(state).detach().numpy()
        # q(s,a) = yhat
        # q(s,a) is [[x, y, z]], the 0 is to get the vector inside
        # the action gets the q(s,a) and updates it with the target (yhat)
        target_full[0, action] = target
        # calls SGD
        self.model.sgd(state, target_full)
        # decay of the explore rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        """[summary]
        Load the previously saved weight, saving the need to retrain the agent
        Args:
            name ([type]): [description] name of the file which store the weights
        """
        self.model.load_weights(name)

    def save(self, name):
        """[summary]
        This save the weights so they can be use later.
        Args:
            name ([type]): [description] filename where the weight will be saved.
        """
        self.model.save_weights(name)


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


def main(data, data_type, train_no=1):
    """[summary]
    This is the main funciton, this runs the training and testing of the agent
    Args:
        data ([type]): Stock data which will be used
        data_type ([type]): test or train
        train_no (int, optional): The number of times the agent is trained choosen by the user.
            Defaults to 1.
    Returns:
            [type]: for training it returns the losses for the training, the cur protfolio,
                dayrewards and final val. For testing it returns the profit from the test,
                protfolio, dayrewards and final val.
    """
    # num_episode = 20
    initial_investment = 20000
    numOfTrain = 2100
    # 2100 train values, 550 test values

    traindata = data[:numOfTrain]
    testdata = data[numOfTrain:]
    # so the index starts at 0
    testdata.reset_index(inplace=True)

    if data_type == "train":
        # training
        env = Env(traindata, initial_investment)
        stateSize = env.state_dim
        action_size = len(env.action_space)
        agent = Agent(stateSize, action_size)
        stand_scaler = get_scaler(env)

        protfolio = []
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
            pickle.dump(stand_scaler, f)

        return agent.model.losses, protfolio, dayRewards, training_val

    elif data_type == "test":
        env = Env(testdata, initial_investment)
        stateSize = env.state_dim
        action_size = len(env.action_space)
        agent = Agent(stateSize, action_size)
        with open(f'scaler.pkl', 'rb') as f:
            stand_scaler = pickle.load(f)

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
    losses, protfolio, day_rewards, training_val = main(data, "train")

    print(f"losses: {losses},\nprotfolio: {protfolio}, \nday_rewards: {day_rewards},\n"
          f"training_val: {training_val}")
    # print(main(data, "test"))
