import gym
import itertools
import numpy as np

from typing import Tuple
from gym import spaces
from collections import namedtuple

# obs_namedtuple = namedtuple('obs_namedtuple',
#                             ['owned', 'open', 'low', 'close', 'high', 'adj_close', 'volume', 'rsi', 'sma', 'ema',
#                              'cash_in_hand'])
obs_namedtuple = namedtuple('obs_namedtuple',
                            ['owned', 'open', 'low', 'close', 'high', 'rsi', 'sma', 'ema', 'cash_in_hand'])
SMA_TIME = 20
EMA_TIME = 20
RSI_TIME = 14


def calculateRsi(data, time=RSI_TIME):
    """
        RSI is an indicator which is used by traders, to look at the strength of
        the current price movement. This returns the RSI value for each state

        Args:
        data ([Dataframe]): The data used to calculate the RSI at a state
        time ([int], optional): RSI calculations use a time frame which it
                                                    spans to get the value. Defaults to rsi_time, which is set to 14.

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

        avgerage_up = up.ewm(span=time, min_periods=time).mean()
        avgerage_down = down.ewm(span=time, min_periods=time).mean()

        rs = abs(avgerage_up / avgerage_down)
        # Change the value of the rsi used to the value displayed on the app
        RSI_TIME = time
        return 100 - (100 / (1 + rs))

    except Exception as e:
        print("Failed! Error", e)


def calculateSMA(data, time=SMA_TIME):
    """
        This calculates the values for the simple moving average.
        Args:
                data ([dataframe]): Data used to calculate the sma
                time (int, optional): This is the time period the moving avergae being
                calculated. Defaults to sma_time, which is set to 20 intially.

        Returns:
                [type]: Moving average values
    """
    global SMA_TIME
    # Change the value of the sma used to the value displayed on the app
    SMA_TIME = time
    return data['Close'].rolling(time).mean()


def calculateEMA(data, time=EMA_TIME):
    """
        This calculates the exponential moving average, this gives more importance
        to the newer data.
        Args:
        data ([dataframe]): Data used to calculate the ema
        time (int, optional): This is the time period the moving avergae being
        calculated. Defaults to ema_time, which is set to 20 intially.

        Returns:
        [type]: [description]
    """
    global EMA_TIME
    # Change the value of the ema used to the value displayed on the app
    EMA_TIME = time
    return data['Close'].ewm(span=time, min_periods=0, adjust=False).mean()


class StockEnv(gym.Env):

    def __init__(self, data, initial_investment=20000):
        super(StockEnv, self).__init__()

        self.stock_price_history = data
        # only working with one stock at a time
        self.n_stock = 1
        # the number of days in the data
        self.n_step = self.stock_price_history.shape[0]
        # instance attributes
        self.obs_namedtuple = None
        self.initial_investment = initial_investment
        self.cur_step = None  # -> the current day in the history
        self.sma = None
        self.ema = None
        self.rsi = None
        self.state_dim = self.n_stock * 8 + 1
        # possibilities -> 3 actions (buy, sell, hold)
        # possibilities = 3^number of stocks
        self.action_space = spaces.Discrete(n=3)
        self.action_list = list(
            map(list, itertools.product([0, 1, 2], repeat=self.n_stock)))
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(self.state_dim, 1), dtype=np.float32)

        # to start the data for day 0 and get all the info for day 0
        self.reset()
        self.post_init()

    def post_init(self):
        owned = 0
        stock_open = self.stock_price_history.Open[0]
        low = self.stock_price_history.Low[0]
        close = self.stock_price_history.Close[0]
        high = self.stock_price_history.High[0]
        rsi = self.sma
        sma = self.rsi
        ema = self.ema
        cash_in_hand = self.initial_investment

        self.obs_namedtuple = obs_namedtuple(owned=owned, open=stock_open, low=low, close=close, high=high, rsi=rsi, sma=sma,
                                             ema=ema, cash_in_hand=cash_in_hand)

    def get_obs(self) -> gym.spaces.MultiDiscrete:
        # todo fix -> obs is not right
        obs = self.reformat_into_matrix(self.obs_namedtuple)

        return obs

    def update_obs(self, owned=None, stock_open=None, low=None, close=None, high=None, adj_close=None, volume=None,
                   rsi=None, sma=None, ema=None, cash_in_hand=None):
        if owned is None:
            owned = self.obs_namedtuple.owned

        if stock_open is None:
            stock_open = self.obs_namedtuple.open

        if low is None:
            low = self.obs_namedtuple.low

        if close is None:
            close = self.obs_namedtuple.close

        if high is None:
            high = self.obs_namedtuple.high

        # if adj_close is None:
        #     adj_close = self.obs_namedtuple.adj_close
        #
        # if volume is None:
        #     volume = self.obs_namedtuple.volume

        if rsi is None:
            rsi = self.obs_namedtuple.rsi

        if sma is None:
            sma = self.obs_namedtuple.sma

        if ema is None:
            ema = self.obs_namedtuple.ema

        if cash_in_hand is None:
            cash_in_hand = self.obs_namedtuple.cash_in_hand

        self.obs_namedtuple = obs_namedtuple(owned=owned, open=stock_open, low=low, close=close, high=high,
                                             rsi=rsi, sma=sma, ema=ema, cash_in_hand=cash_in_hand)

    def step(self, action) -> Tuple[gym.spaces.MultiDiscrete, float, bool, dict]:
        reward = 0

        cur_reward = 0
        done = False
        episode_done = False
        prev_reward = self.get_reward()

        stock_owned = self.obs_namedtuple.owned
        cash_in_hand = self.initial_investment
        stock_open = self.stock_price_history.Open[self.cur_step]
        stock_low = self.stock_price_history.Low[self.cur_step]
        stock_close = self.stock_price_history.Close[self.cur_step]
        stock_high = self.stock_price_history.High[self.cur_step]
        # stock_adj_close = self.stock_price_history['Adj Close'][self.cur_step]
        # stock_vol = self.stock_price_history.Volume[self.cur_step]
        # indicators
        # rsi ema and sma has nan for the first couple of values.
        self.rsi = calculateRsi(self.stock_price_history)
        if np.isnan(self.rsi[self.cur_step]):
            stock_rsi = 0
        else:
            stock_rsi = self.rsi[self.cur_step]

        self.sma = calculateSMA(self.stock_price_history)
        self.ema = calculateEMA(self.stock_price_history)
        # as this use previous data to get the current value, this can be nan
        # at the first few time steps, this is to remove them
        if np.isnan(self.sma[self.cur_step]):
            stock_sma = stock_close
        else:
            stock_sma = self.sma[self.cur_step]

        if np.isnan(self.ema[self.cur_step]):
            stock_ema = stock_close
        else:
            stock_ema = self.ema[self.cur_step]

        self.obs_namedtuple = obs_namedtuple(owned=stock_owned, open=stock_open, low=stock_low, close=stock_close,
                                             high=stock_high, rsi=stock_rsi, sma=stock_sma, ema=stock_ema,
                                             cash_in_hand=cash_in_hand)
        self.trade(action)

        cur_reward = self.get_reward()
        reward = cur_reward - prev_reward

        # go to the next step
        self.cur_step += 1

        # episode length is 28days  (4 weeks or 1 month)
        if self.cur_step % 28 == 0:
            episode_done = True

        # done if the data is finished
        done = self.cur_step == self.n_step - 1

        info = {'cur_reward': cur_reward}

        return self.get_obs(), reward, done, info

    def get_reward(self):

        reward = (self.obs_namedtuple.owned * self.obs_namedtuple.close) + self.obs_namedtuple.cash_in_hand

        return reward

    def reset(self) -> gym.spaces.MultiDiscrete:
        self.cur_step = 0
        stock_owned = np.zeros(self.n_stock)

        cash_in_hand = self.initial_investment
        stock_open = self.stock_price_history.Open[self.cur_step]
        stock_low = self.stock_price_history.Low[self.cur_step]
        stock_close = self.stock_price_history.Close[self.cur_step]
        stock_high = self.stock_price_history.High[self.cur_step]
        # stock_adj_close = self.stock_price_history['Adj Close'][self.cur_step]
        # stock_vol = self.stock_price_history.Volume[self.cur_step]
        # indicators
        # rsi ema and sma has nan for the first couple of values.
        self.rsi = calculateRsi(self.stock_price_history)
        if np.isnan(self.rsi[self.cur_step]):
            stock_rsi = 0
        else:
            stock_rsi = self.rsi[self.cur_step]

        self.sma = calculateSMA(self.stock_price_history)
        self.ema = calculateEMA(self.stock_price_history)
        # as this use previous data to get the current value, this can be nan
        # at the first few time steps, this is to remove them
        if np.isnan(self.sma[self.cur_step]):
            stock_sma = stock_close
        else:
            stock_sma = self.sma[self.cur_step]

        if np.isnan(self.ema[self.cur_step]):
            stock_ema = stock_close
        else:
            stock_ema = self.ema[self.cur_step]

        observation_namedtuple = obs_namedtuple(owned=stock_owned, open=stock_open, low=stock_low, close=stock_close,
                                                high=stock_high, rsi=stock_rsi, sma=stock_sma, ema=stock_ema,
                                                cash_in_hand=cash_in_hand)

        obs = self.reformat_into_matrix(observation_namedtuple)

        return obs

    def reformat_into_matrix(self, observation_namedtuple):

        stock_owned = observation_namedtuple.owned
        cash_in_hand = observation_namedtuple.cash_in_hand
        stock_open = observation_namedtuple.open
        stock_low = observation_namedtuple.low
        stock_close = observation_namedtuple.close
        stock_high = observation_namedtuple.high
        # stock_adj_close = observation_namedtuple.adj_close
        # stock_vol = observation_namedtuple.volume
        stock_vol = 000000000
        stock_rsi = observation_namedtuple.rsi
        stock_sma = observation_namedtuple.sma
        stock_ema = observation_namedtuple.ema

        norm_stock_owned = np.interp(stock_owned, [0, stock_vol], [0.0, 1.0]).reshape(1, 1)
        norm_cash_in_hand = np.interp(cash_in_hand, [0, stock_vol*stock_close], [0.0, 1.0]).reshape(1, 1)
        norm_stock_open = np.interp(stock_open, [0, self.stock_price_history['High'].max()*1.1],
                                    [0.0, 1.0]).reshape(1, 1)
        norm_stock_low = np.interp(stock_low, [0, self.stock_price_history['High'].max()*1.1],
                                   [0.0, 1.0]).reshape(1, 1)
        norm_stock_close = np.interp(stock_close, [0, self.stock_price_history['High'].max()*1.1],
                                     [0.0, 1.0]).reshape(1, 1)
        norm_stock_high = np.interp(stock_high, [0, self.stock_price_history['High'].max()*1.1],
                                    [0.0, 1.0]).reshape(1, 1)
        # norm_stock_adj_close = np.interp(stock_adj_close, [0, self.stock_price_history['High'].max() * 1.1],
        #                                  [0.0, 1.0]).reshape(1, 1)
        # norm_stock_vol = np.interp(stock_vol, [0, self.stock_price_history['Volume'].max() * 1.1],
        #                            [0.0, 1.0]).reshape(1, 1)
        norm_stock_rsi = np.interp(stock_rsi, [0, self.rsi.max() * 1.1], [0.0, 1.0]).reshape(1, 1)
        norm_stock_sma = np.interp(stock_sma, [0, self.sma.max() * 1.1], [0.0, 1.0]).reshape(1, 1)
        norm_stock_ema = np.interp(stock_ema, [0, self.ema.max() * 1.1], [0.0, 1.0]).reshape(1, 1)

        row_matrix = np.concatenate((norm_stock_owned, norm_stock_open, norm_stock_high, norm_stock_low,
                                     norm_stock_close, norm_stock_rsi, norm_stock_sma, norm_stock_ema,
                                     norm_cash_in_hand))

        return row_matrix

    def trade(self, action):

        action_vec = self.action_space.sample()
        cash_in_hand = self.obs_namedtuple.cash_in_hand

        allocations = []
        sell_index = []
        buy_index = []

        if action_vec == 0:
            # sell stocks
            sell_index.append(1)
        elif action_vec == 2:
            # buy stocks
            buy_index.append(1)

        if sell_index:
            cash_in_hand += self.obs_namedtuple.close * self.obs_namedtuple.owned
            owned = 0
            self.update_obs(cash_in_hand=cash_in_hand, owned=owned)
        #     make update obs_namedtyple function
        elif buy_index:
            can_buy = True
            while can_buy:
                owned = self.obs_namedtuple.owned
                for i in buy_index:
                    # buy a single stock on loop
                    # if there are multiple stocks then the loop iterates over each stock and buys 1 stock
                    # (if it can) for the stocks available each iteration.
                    if self.obs_namedtuple.cash_in_hand > self.obs_namedtuple.close:
                        owned += 1  # buy one share
                        cash_in_hand -= self.obs_namedtuple.close
                        self.update_obs(owned=owned, cash_in_hand=cash_in_hand)
                    else:
                        # if the cash left is lower the stock price then stop buying
                        can_buy = False
        return allocations

    def render(self, mode="human") -> None:
        pass

    def close (self) -> None:
        pass