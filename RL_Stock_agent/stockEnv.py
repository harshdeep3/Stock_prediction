import gymnasium as gym
from gymnasium import spaces
from collections import namedtuple
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
import logging

import MT5_Link as link
import MetaTrader5 as mt5
import numpy as np
import pandas as pd
import pandas_datareader as web
import datetime
import yfinance as yf
yf.pdr_override()

from ta.trend import sma_indicator
from ta.trend import ema_indicator
from ta.momentum import RSIIndicator


obs_namedtuple = namedtuple('obs_namedtuple',
                            ['owned', 'open', 'low', 'close', 'high', 'volume', 'sma', 'ema', 'rsi', 'cash_in_hand'])

SMA_TIME = 20
EMA_TIME = 20
RSI_TIME = 14

LOGGING_LEVEL = logging.DEBUG

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


class StockMarketEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, stock_data, cash_in_hand=400):
        super(StockMarketEnv, self).__init__()
                
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(LOGGING_LEVEL)     

        # Define action and observation spaces
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=0, high=1, shape=(10,), dtype=np.float64)

        # Initialize stock data
        self.data = stock_data
        self.data["pct_change"] = self.data["close"].pct_change()
        self.data = self.data.dropna()

        # Initialize state and reward
        self.state = None
        self.reward = 0
        self.cash_in_hand = cash_in_hand
        self.total_net = cash_in_hand
        self.total_list = []

        # get indicator data
        self.rsi = RSIIndicator(self.data['close'],window=RSI_TIME, fillna=True).rsi()
        self.sma = sma_indicator(self.data['close'],window=SMA_TIME, fillna=True)
        self.ema = ema_indicator(self.data['close'],window=EMA_TIME, fillna=True)

        # Set starting index for data
        self.current_index = 1
        self.reset()

    def step(self, action):
        
        # Get current price and calculate reward
        current_price = self.data["close"].iloc[self.current_index]
        self.reward = 0 if action == 0 else current_price * 0.01
        # Update state
        stock_owned = self.state.owned
        cash_in_hand = self.cash_in_hand
        stock_open = self.data.open.iloc[self.current_index]
        stock_low = self.data.low.iloc[self.current_index]
        stock_close = self.data.close.iloc[self.current_index]
        stock_high = self.data.high.iloc[self.current_index]
        stock_vol = self.data.tick_volume.iloc[self.current_index]
        stock_rsi = 0
        stock_sma = stock_close
        stock_ema = stock_close
        # self.state[-1] = self.data.iloc[self.current_index]["pct_change"]
        if not np.isnan(self.rsi.iloc[self.current_index]):
            stock_rsi = self.rsi.iloc[self.current_index]

        # as this use previous data to get the current value, this can be nan
        # at the first few time steps, this is to remove them
        if not np.isnan(self.sma.iloc[self.current_index]):
            stock_sma = self.sma.iloc[self.current_index]

        if not np.isnan(self.ema.iloc[self.current_index]):
            stock_ema = self.ema.iloc[self.current_index]

        self.state = obs_namedtuple(owned=stock_owned, open=stock_open, low=stock_low, close=stock_close,
                                    high=stock_high, volume=stock_vol, sma=stock_sma, ema=stock_ema, rsi=stock_rsi,
                                    cash_in_hand=cash_in_hand)
        self.trade(action)
        # Move to next time step
        self.current_index += 1
        if self.current_index == 13499:
            self.reset()

        self.logger.debug(f"action: {action}, stocks owned: {stock_owned}, cash in hand: {cash_in_hand}, open: {stock_open}, low:" 
                          f"{stock_low}, close: {stock_close}, high: {stock_high}, volume: {stock_vol}, RSI: {stock_rsi},"
                          f"SMA: {stock_sma}, EMA: {stock_ema}")
        
        # Check if done
        done = self.current_index == len(self.data)
        reward = self.get_reward()
        self.total_net = self.get_net()
        # Return observation, reward, done, and info
        return self.reformat_matrix(self.state), reward, done, False, {}

    def trade(self, action):
        
        cash_in_hand = self.cash_in_hand

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
            # make update obs_namedtyple function
            self.update_obs(cash_in_hand=cash_in_hand, owned=owned)
            
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
        
        self.cash_in_hand = cash_in_hand
        
        self.logger.debug(f"action: {action}, cash in hand: {cash_in_hand}, state cash in hand: {self.state.cash_in_hand}, owned: {self.state.owned}")

    def get_reward(self):
        self.logger.debug("get_reward function - > ")

        print(f"index: {self.current_index}, len of data.close: {len(self.data['close'])}")
        reward = (self.state.owned * self.data['close'].iloc[self.current_index]) + self.cash_in_hand

        return reward
    
    def get_net(self):
        self.logger.debug("get_net function - > ")
        
        total_net = (self.state.owned * self.data.close.iloc[self.current_index]) + self.cash_in_hand

        return total_net

    def reset(self, seed=None):        
        # Reset state, reward, and current index
        self.reward = 0
        self.current_index = 1

        info = {}

        stock_owned = 0
        
        stock_open = self.data['open'].iloc[self.current_index]
        stock_high = self.data['high'].iloc[self.current_index]
        stock_low = self.data['low'].iloc[self.current_index]
        stock_close = self.data['close'].iloc[self.current_index]
        stock_vol = self.data['tick_volume'].iloc[self.current_index]
        cash_in_hand = 200
        
        #todo add % change
        # self.state[-1] = self.data.iloc[self.current_index]["pct_change"]

        stock_rsi = self.rsi.iloc[self.current_index]

        # as this use previous data to get the current value, this can be nan
        # at the first few time steps, this is to remove them
        stock_sma = self.sma.iloc[self.current_index]

        stock_ema = self.ema.iloc[self.current_index]

        self.state = obs_namedtuple(owned=stock_owned, open=stock_open, low=stock_low, close=stock_close,
                                    high=stock_high, volume=stock_vol, sma=stock_sma, ema=stock_ema, rsi=stock_rsi,
                                    cash_in_hand=cash_in_hand)
        
        self.total_net = self.get_net()
        state_output = self.reformat_matrix(self.state)
        self.logger.debug(f"stocks owned: {stock_owned}, cash in hand: {cash_in_hand}, open: {stock_open}, low:"
                        f"{stock_low}, close: {stock_close}, high: {stock_high}, volume: {stock_vol}, RSI: {stock_rsi},"
                        f"SMA: {stock_sma}, EMA: {stock_ema}")
        return state_output, info

    def reformat_matrix(self, state):
        
        owned = state.owned
        price_open = state.open
        high = state.high
        low = state.low
        close = state.close
        volume = state.volume
        rsi = state.rsi
        sma = state.sma
        ema = state.ema
        cash_in_hand = state.cash_in_hand

        norm_stock_owned = np.interp(owned, [0, volume], [0.0, 1.0]).reshape(1, 1)
        norm_cash_in_hand = np.interp(cash_in_hand, [0, volume*close], [0.0, 1.0]).reshape(1, 1)
        norm_stock_open = np.interp(price_open, [0, self.data['open'].max()*1.1],
                                    [0.0, 1.0]).reshape(1, 1)
        norm_stock_low = np.interp(low, [0, self.data['low'].max()*1.1],
                                   [0.0, 1.0]).reshape(1, 1)
        norm_stock_close = np.interp(close, [0, self.data['close'].max()*1.1],
                                     [0.0, 1.0]).reshape(1, 1)
        norm_stock_high = np.interp(high, [0, self.data['high'].max()*1.1],
                                    [0.0, 1.0]).reshape(1, 1)
        norm_stock_vol = np.interp(volume, [0, self.data['tick_volume'].max() * 1.1],
                                   [0.0, 1.0]).reshape(1, 1)
        norm_stock_sma = np.interp(sma, [0, self.sma.max() * 1.1],
                                   [0.0, 1.0]).reshape(1, 1)
        norm_stock_ema = np.interp(ema, [0, self.ema.max() * 1.1],
                                   [0.0, 1.0]).reshape(1, 1)
        norm_stock_rsi = np.interp(rsi, [0, self.rsi.max() * 1.1],
                                   [0.0, 1.0]).reshape(1, 1)

        row_matrix = np.concatenate((norm_stock_owned, norm_stock_open, norm_stock_high, norm_stock_low,
                                     norm_stock_close, norm_stock_vol, norm_stock_sma, norm_stock_ema, norm_stock_rsi,
                                     norm_cash_in_hand)).reshape(10,)

        self.logger.debug(f"stocks owned: {norm_stock_owned}, cash in hand: {norm_cash_in_hand}, open: {norm_stock_open}, low: "
                          f"{norm_stock_low}, close: {norm_stock_close}, high: {norm_stock_high}, volume: {norm_stock_vol}, RSI: {norm_stock_rsi},"
                          f"SMA: {norm_stock_sma}, EMA: {norm_stock_ema}")
        
        return row_matrix

    def get_obs(self):        
        
        obs = self.reformat_matrix(self.state)
        
        self.logger.debug(f"obs: {obs}")

        return obs

    def update_obs(self, owned=None, stock_open=None, low=None, close=None, high=None, volume=None,
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

        if volume is None:
            volume = self.state.volume

        if rsi is None:
            if not np.isnan(self.state.rsi):
                rsi = self.state.rsi

        if sma is None:
            if not np.isnan(self.state.sma):
                sma = self.state.sma

        if ema is None:
            if not np.isnan(self.state.ema):
                ema = self.state.ema

        if cash_in_hand is None:
            cash_in_hand = self.state.cash_in_hand

        self.logger.debug(f"stocks owned: {owned}, cash in hand: {cash_in_hand}, open: {stock_open}, low:"
                        f"{low}, close: {close}, high: {high}, volume: {volume}, RSI: {rsi},"
                        f"SMA: {sma}, EMA: {ema}")
        
        self.state = obs_namedtuple(owned=owned, open=stock_open, low=low, close=close, high=high,
                                    volume=volume, sma=sma, ema=ema, rsi=rsi, cash_in_hand=cash_in_hand)


if __name__ == '__main__':
    # MT5 account connection
    mt5_obj = link.MT5Class()
    mt5_obj.login_to_metatrader()
    mt5_obj.get_acc_info()

    start = datetime.datetime(2010, 7, 1).strftime("%Y-%m-%d")
    end = datetime.datetime.now().strftime("%Y-%m-%d")

    timeframe_h1 = mt5.TIMEFRAME_H1
    timeframe_h4 = mt5.TIMEFRAME_H4
    timeframe_d1 = mt5.TIMEFRAME_D1
    timeframe_w1 = mt5.TIMEFRAME_W1
    symbol = 'USDJPY'
    count = 8500  # get 8500 data points

    data_h1 = link.get_historic_data(fx_symbol=symbol, fx_timeframe=timeframe_h1, fx_count=count)
    # data_h4 = link.get_historic_data(fx_symbol=symbol, fx_timeframe=timeframe_h4, fx_count=count)
    # data_d1 = link.get_historic_data(fx_symbol=symbol, fx_timeframe=timeframe_d1, fx_count=count)
    # data_w1 = link.get_historic_data(fx_symbol=symbol, fx_timeframe=timeframe_w1, fx_count=count)

    # print("H\n", data_h1.head())
    # print("\nH4\n", data_h4.head())
    # print("\nD1\n", data_d1.head())
    # print("\nW1\n", data_w1.head())

    if data_h1 is None:
        print("Error: Data not recieved!")
    else:
        data_h1.set_index('time', inplace=True)
        print("\n", data_h1.head())
        print("\n", data_h1.tail())
        # env = StockMarketEnv(data_h1)

        # model = DQN('MlpPolicy', env, verbose=1)
        # model.learn(total_timesteps=1000)
