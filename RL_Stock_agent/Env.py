import gym
import itertools
import numpy as np

from gym import spaces
from collections import namedtuple

obs_namedtuple = namedtuple('obs_namedtuple',
                            ['owned', 'open', 'low', 'close', 'high', 'adj_close', 'volume', 'rsi', 'sma', 'ema',
                             'cash_in_hand'])

sma_time = 20
ema_time = 20
rsi_time = 14

def calculateRsi(data, time=rsi_time):
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

        avgerage_up = up.ewm(span=time, min_periods=time).mean()
        avgerage_down = down.ewm(span=time, min_periods=time).mean()

        rs = abs(avgerage_up / avgerage_down)
        # Change the value of the rsi used to the value displayed on the app
        rsi_time = time
        return 100 - (100 / (1 + rs))

    except Exception as e:
        print("Failed! Error", e)


def calculateSMA(data, time=sma_time):
    """
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


def calculateEMA(data, time=ema_time):
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
    global ema_time
    # Change the value of the ema used to the value displayed on the app
    ema_time = time
    return data['Close'].ewm(span=time, min_periods=0, adjust=False).mean()


class StockEnv(gym.Env):

    def __init__(self, data, initial_investment=20000):
        super(StockEnv, self).__init__()

        # getting the data and using it to get the number of day in the history and the number of stock
        self.stock_price_history = data
        # only working with one stock at a time
        self.n_stock = 1
        # the number of days in the data
        self.n_step = self.stock_price_history.shape[0]
        # instance attributes
        self.initial_investment = initial_investment
        self.cur_step = None  # -> the current day in the history
        self.obs_namedtuple = None
        self.sma = None
        self.ema = None
        self.rsi = None

        self.state_dim = self.n_stock * 10 + 1
        # possibilities -> 3 actions (buy, sell, hold)
        self.action = None
        # possibilities = 3^number of stocks
        self.action_space = spaces.Discrete(n=3)
        self.obs_space = spaces.Box(low=0.0, high=1.0, shape=(self.state_dim, 1))

        # to start the data for day 0 and get all the info for day 0
        self.reset()

    def reset(self):
        self.cur_step = 0
        stock_owned = np.zeros(self.n_stock)

        cash_in_hand = self.initial_investment
        stock_open = self.stock_price_history.Open[self.cur_step]
        stock_low = self.stock_price_history.Low[self.cur_step]
        stock_close = self.stock_price_history.Close[self.cur_step]
        stock_high = self.stock_price_history.High[self.cur_step]
        stock_adj_close = self.stock_price_history['Adj Close'][self.cur_step]
        stock_vol = self.stock_price_history.Volume[self.cur_step]
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
                                                high=stock_high, adj_close=stock_adj_close, volume=stock_vol,
                                                rsi=stock_rsi, sma=stock_sma, ema=stock_ema, cash_in_hand=cash_in_hand)

        obs = self.reformat_into_matrix(observation_namedtuple)

        return obs

    def reformat_into_matrix(self, observation_namedtuple):

        stock_owned = observation_namedtuple.owned
        cash_in_hand = observation_namedtuple.cash_in_hand
        stock_open = observation_namedtuple.open
        stock_low = observation_namedtuple.low
        stock_close = observation_namedtuple.close
        stock_high = observation_namedtuple.high
        stock_adj_close = observation_namedtuple.adj_close
        stock_vol = observation_namedtuple.volume
        stock_rsi = observation_namedtuple.rsi
        stock_sma = observation_namedtuple.sma
        stock_ema = observation_namedtuple.ema

        self.obs_space = observation_namedtuple

        norm_stock_owned = np.interp(stock_owned, [0, stock_vol], [0.0, 1.0]).reshape(1,1)
        norm_cash_in_hand = np.interp(cash_in_hand, [0, stock_vol*stock_close], [0.0, 1.0]).reshape(1,1)
        norm_stock_open = np.interp(stock_open, [0, self.stock_price_history['High'].max()*1.1],
                                    [0.0, 1.0]).reshape(1,1)
        norm_stock_low = np.interp(stock_low, [0, self.stock_price_history['High'].max()*1.1],
                                   [0.0, 1.0]).reshape(1,1)
        norm_stock_close = np.interp(stock_close, [0, self.stock_price_history['High'].max()*1.1],
                                     [0.0, 1.0]).reshape(1,1)
        norm_stock_high = np.interp(stock_high, [0, self.stock_price_history['High'].max()*1.1],
                                    [0.0, 1.0]).reshape(1,1)
        norm_stock_adj_close = np.interp(stock_adj_close, [0, self.stock_price_history['High'].max() * 1.1],
                                         [0.0, 1.0]).reshape(1,1)
        norm_stock_vol = np.interp(stock_vol, [0, self.stock_price_history['Volume'].max() * 1.1],
                                   [0.0, 1.0]).reshape(1,1)
        norm_stock_rsi = np.interp(stock_rsi, [0, self.rsi.max() * 1.1], [0.0, 1.0]).reshape(1,1)
        norm_stock_sma = np.interp(stock_sma, [0, self.sma.max() * 1.1], [0.0, 1.0]).reshape(1,1)
        norm_stock_ema = np.interp(stock_ema, [0, self.ema.max() * 1.1], [0.0, 1.0]).reshape(1,1)

        row_matrix = np.concatenate((norm_stock_owned, norm_stock_open, norm_stock_high, norm_stock_low,
                                     norm_stock_close, norm_stock_adj_close, norm_stock_vol, norm_stock_rsi,
                                     norm_stock_sma, norm_stock_ema, norm_cash_in_hand))

        return row_matrix

    def step(self, action):
        pass

    def get_obs(self):
        pass

    def get_val(self):
        pass

    def render(self, mode="human"):
        pass

    def close(self):
        pass
