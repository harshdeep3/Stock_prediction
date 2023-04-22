from gym import spaces
from collections import namedtuple
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env

import gym
import numpy as np
import pandas_datareader as web
import datetime
import yfinance as yf
yf.pdr_override()


obs_namedtuple = namedtuple('obs_namedtuple',
                            ['owned', 'open', 'low', 'close', 'high', 'adj_close', 'volume', 'cash_in_hand'])

SMA_TIME = 20
EMA_TIME = 20
RSI_TIME = 14


def get_data(stock_name, start_date, end_date):
    """[summary]
    This gets the data which is used by the agent.
    Returns:
        dataframe [dataframe]: This returns the dataframe which is used for the state and by the
        agent
    """
    df = web.data.get_data_yahoo(stock_name, start_date, end_date)
    # reset the index of the data to normal ints so df['Date'] can be used
    df.reset_index()
    return df


def calculate_rsi(stock_data, time=RSI_TIME):
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
        diff = stock_data.Close.diff()
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
        return 100 - (100 / (1 + rs))

    except Exception as e:
        print("Failed! Error", e)


def calculate_sma(stock_data, time=SMA_TIME):
    """
        This calculates the values for the simple moving average.
        Args:
                stock_data ([dataframe]): Data used to calculate the sma
                time (int, optional): This is the time period the moving avergae being
                calculated. Defaults to sma_time, which is set to 20 intially.

        Returns:
                [type]: Moving average values
    """
    global SMA_TIME
    return stock_data['Close'].rolling(time).mean()


def calculate_ema(stock_data, time=EMA_TIME):
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

    return stock_data['Close'].ewm(span=time, min_periods=0, adjust=False).mean()


class StockMarketEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, stock_data, cash_in_hand=20000):
        super(StockMarketEnv, self).__init__()

        # Define action and observation spaces
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=0, high=1, shape=(8,), dtype=np.float64)

        # Initialize stock data
        self.data = stock_data
        self.data["pct_change"] = self.data["Close"].pct_change()
        self.data = self.data.dropna()

        # Initialize state and reward
        self.state = None
        self.reward = 0
        self.cash_in_hand = cash_in_hand

        # Set starting index for data
        self.current_index = 0
        self.reset()

    def step(self, action):
        reward = 0

        cur_reward = 0
        done = False
        episode_done = False
        prev_reward = self.get_reward()

        # Get current price and calculate reward
        current_price = self.data["Close"][self.current_index]
        self.reward = 0 if action == 0 else current_price * 0.01

        # Update state
        stock_owned = self.state.owned
        cash_in_hand = self.cash_in_hand
        stock_open = self.data.Open[self.current_index]
        stock_low = self.data.Low[self.current_index]
        stock_close = self.data.Close[self.current_index]
        stock_high = self.data.High[self.current_index]
        stock_adj_close = self.data['Adj Close'][self.current_index]
        stock_vol = self.data.Volume[self.current_index]

        # self.state[-1] = self.data.iloc[self.current_index]["pct_change"]

        self.state = obs_namedtuple(owned=stock_owned, open=stock_open, low=stock_low, close=stock_close,
                                    high=stock_high, adj_close=stock_adj_close, volume=stock_vol,
                                    cash_in_hand=cash_in_hand)
        self.trade(action)

        # Move to next time step
        self.current_index += 1

        # Check if done
        done = self.current_index == len(self.data)

        # Return observation, reward, done, and info
        return self.reformat_matrix(self.state), self.reward, done, {}

    def trade(self, action):

        cash_in_hand = self.state.cash_in_hand

        allactions = []
        sell_index = []
        buy_index = []

        if action == 0:
            # sell stocks
            sell_index.append(1)
        elif action == 2:
            # buy stocks
            buy_index.append(1)

        if sell_index:
            cash_in_hand += self.state.close * self.state.owned
            owned = 0
            self.update_obs(cash_in_hand=cash_in_hand, owned=owned)
        #     make update obs_namedtyple function
        elif buy_index:
            can_buy = True
            while can_buy:
                owned = self.state.owned
                for _ in buy_index:
                    # buy a single stock on loop
                    # if there are multiple stocks then the loop iterates over each stock and buys 1 stock
                    # (if it can) for the stocks available each iteration.
                    if self.state.cash_in_hand > self.state.close:
                        owned += 1  # buy one share
                        cash_in_hand -= self.state.close
                        self.update_obs(owned=owned, cash_in_hand=cash_in_hand)
                    else:
                        # if the cash left is lower the stock price then stop buying
                        can_buy = False
        return allactions

    def get_reward(self):

        reward = (self.state.owned * self.state.close) + self.state.cash_in_hand

        return reward

    def reset(self):
        # Reset state, reward, and current index
        self.reward = 0
        self.current_index = 0

        owned = 0
        price_open = self.data['Open'][self.current_index]
        high = self.data['High'][self.current_index]
        low = self.data['Low'][self.current_index]
        close = self.data['Close'][self.current_index]
        adj_close = self.data['Adj Close'][self.current_index]
        volume = self.data['Volume'][self.current_index]
        cash_in_hand = 20000

        self.state = obs_namedtuple(owned=owned, open=price_open, high=high, low=low, close=close, adj_close=adj_close,
                                    volume=volume, cash_in_hand=cash_in_hand)

        return self.reformat_matrix(self.state)

    def reformat_matrix(self, state):

        owned = state.owned
        price_open = state.open
        high = state.high
        low = state.low
        close = state.close
        adj_close = state.adj_close
        volume = state.volume
        cash_in_hand = state.cash_in_hand

        norm_stock_owned = np.interp(owned, [0, volume], [0.0, 1.0]).reshape(1, 1)
        norm_cash_in_hand = np.interp(cash_in_hand, [0, volume*close], [0.0, 1.0]).reshape(1, 1)
        norm_stock_open = np.interp(price_open, [0, self.data['Open'].max()*1.1],
                                    [0.0, 1.0]).reshape(1, 1)
        norm_stock_low = np.interp(low, [0, self.data['Low'].max()*1.1],
                                   [0.0, 1.0]).reshape(1, 1)
        norm_stock_close = np.interp(close, [0, self.data['Close'].max()*1.1],
                                     [0.0, 1.0]).reshape(1, 1)
        norm_stock_high = np.interp(high, [0, self.data['High'].max()*1.1],
                                    [0.0, 1.0]).reshape(1, 1)
        norm_stock_adj_close = np.interp(adj_close, [0, self.data['Adj Close'].max() * 1.1],
                                         [0.0, 1.0]).reshape(1, 1)
        norm_stock_vol = np.interp(volume, [0, self.data['Volume'].max() * 1.1],
                                   [0.0, 1.0]).reshape(1, 1)

        row_matrix = np.concatenate((norm_stock_owned, norm_stock_open, norm_stock_high, norm_stock_low,
                                     norm_stock_close, norm_stock_adj_close, norm_stock_vol,
                                     norm_cash_in_hand)).reshape(8,)

        return row_matrix

    def get_obs(self):
        obs = self.reformat_matrix(self.state)

        return obs

    def update_obs(self, owned=None, stock_open=None, low=None, close=None, high=None, adj_close=None, volume=None,
                   rsi=None, sma=None, ema=None, cash_in_hand=None):
        if owned is None:
            owned = self.state.owned

        if stock_open is None:
            stock_open = self.state.open

        if low is None:
            low = self.state.low

        if close is None:
            close = self.state.close

        if high is None:
            high = self.state.high

        if adj_close is None:
            adj_close = self.state.adj_close

        if volume is None:
            volume = self.state.volume

        # if rsi is None:
        #     rsi = self.state.rsi
        #
        # if sma is None:
        #     sma = self.state.sma
        #
        # if ema is None:
        #     ema = self.state.ema

        if cash_in_hand is None:
            cash_in_hand = self.state.cash_in_hand

        self.state = obs_namedtuple(owned=owned, open=stock_open, low=low, close=close, high=high,
                                    adj_close=adj_close, volume=volume, cash_in_hand=cash_in_hand)

    def render(self, mode='human'):
        pass


if __name__ == '__main__':
    start = datetime.datetime(2010, 7, 1).strftime("%Y-%m-%d")
    end = datetime.datetime.now().strftime("%Y-%m-%d")
    data = get_data("AAPL", start, end)

    print(data.min())
    print(data['Adj Close'][0])
    env = StockMarketEnv(data)

    model = DQN('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=10000)
