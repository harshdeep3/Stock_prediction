import numpy as np
import itertools

SMA_TIME = 20
EMA_TIME = 20
RSI_TIME = 14

def calculateRsi(data, time=RSI_TIME):
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

        rs = abs(avgerageUp /avgerageDown)
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
            time (int, optional): This is the time period the moving avergae being
            calculated. Defaults to sma_time, which is set to 20 intially.

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
            data ([dataframe]): Data used to calculate the ema
            time (int, optional): This is the time period the moving avergae being
            calculated. Defaults to ema_time, which is set to 20 intially.

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
        ####################################################################
        ######################## state size of 3 ##########################
        if self.state_size_used == 3 :
            self.stock_price = None

            self.state_dim = self.n_stock * 2 + 1
        ####################################################################

        ####################################################################
        ######################## state size of 4 ##########################
        elif self.state_size_used == 4:
            self.stock_open = None
            self.stock_close = None

            self.state_dim = self.n_stock * 3 + 1
        ####################################################################

        ####################################################################
        ######################## state size of 5 ##########################
        elif self.state_size_used == 5:
            self.stock_open = None
            self.stock_close = None
            self.stock_Volume = None

            self.state_dim = self.n_stock * 4 + 1
        ####################################################################

        ####################################################################
        ######################## state size of 6 ##########################
        elif self.state_size_used == 6:
            self.stock_open = None
            self.stock_high = None
            self.stock_low = None
            self.stock_close = None

            self.state_dim = self.n_stock * 5 + 1
        ####################################################################

        ####################################################################
        ######################## state size of 7 ##########################
        elif self.state_size_used == 7:
            self.stock_open = None
            self.stock_high = None
            self.stock_low = None
            self.stock_close = None
            self.stock_Volume = None

            self.state_dim = self.n_stock * 6 + 1
        ####################################################################

        ####################################################################
        ######################## state size of 8 ##########################
        elif self.state_size_used == 8:
            self.stock_open = None
            self.stock_high = None
            self.stock_low = None
            self.stock_close = None
            self.stock_Volume = None
            self.stock_AdjClose = None

            self.state_dim = self.n_stock * 7 + 1
        ####################################################################

        ####################################################################
        ######################## state size of 9 ##########################
        elif self.state_size_used == 9:
            self.stock_open = None
            self.stock_high = None
            self.stock_low = None
            self.stock_close = None

            self.stock_AdjClose = None
            self.stock_Volume = None
            self.sma = None

            self.state_dim = self.n_stock * 8 + 1
        ####################################################################

        ####################################################################
        ######################## state size of 11 ##########################
        elif self.state_size_used == 11:
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
        ####################################################################

        self.cash_in_hand = None  # -> cash left after invesment
        # possibilities of the action -> 3 actions (buy, sell, hold)
        # possibilities = 3^number of stocks
        self.action_space = np.arange( 3**self.n_stock)

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
        ####################################################################
        ######################## state size of 3 ##########################
        if self.state_size_used == 3:
            self.stock_price = self.stock_price_history.Close[self.cur_step]

        ####################################################################

        ####################################################################
        ######################## state size of 4 ##########################
        elif self.state_size_used == 4:
            self.stock_open = self.stock_price_history.Open[self.cur_step]
            self.stock_close = self.stock_price_history.Close[self.cur_step]
        ####################################################################

        ####################################################################
        ######################## state size of 5 ##########################
        elif self.state_size_used == 5:
            self.stock_open = self.stock_price_history.Open[self.cur_step]
            self.stock_close = self.stock_price_history.Close[self.cur_step]
            self.stock_Volume = self.stock_price_history.Volume[self.cur_step]

        ####################################################################

        ####################################################################
        ######################## state size of 6 ##########################
        elif self.state_size_used == 6:
            self.stock_open = self.stock_price_history.Open[self.cur_step]
            self.stock_high = self.stock_price_history.High[self.cur_step]
            self.stock_low = self.stock_price_history.Low[self.cur_step]
            self.stock_close = self.stock_price_history.Close[self.cur_step]

        ####################################################################

        ####################################################################
        ######################## state size of 7 ##########################
        elif self.state_size_used == 7:
            self.stock_open = self.stock_price_history.Open[self.cur_step]
            self.stock_high = self.stock_price_history.High[self.cur_step]
            self.stock_low = self.stock_price_history.Low[self.cur_step]
            self.stock_close = self.stock_price_history.Close[self.cur_step]
            self.stock_Volume = self.stock_price_history.Volume[self.cur_step]

        ####################################################################

        ####################################################################
        ######################## state size of 8 ###########################
        elif self.state_size_used == 8:
            self.stock_open = self.stock_price_history.Open[self.cur_step]
            self.stock_high = self.stock_price_history.High[self.cur_step]
            self.stock_low = self.stock_price_history.Low[self.cur_step]
            self.stock_close = self.stock_price_history.Close[self.cur_step]
            self.stock_AdjClose = self.stock_price_history['Adj Close'][self.cur_step]
            self.stock_Volume = self.stock_price_history.Volume[self.cur_step]

        ####################################################################

        ####################################################################
        ######################## state size of 9 ###########################
        elif self.state_size_used == 9:
            self.stock_open = self.stock_price_history.Open[self.cur_step]
            self.stock_high = self.stock_price_history.High[self.cur_step]
            self.stock_low = self.stock_price_history.Low[self.cur_step]
            self.stock_close = self.stock_price_history.Close[self.cur_step]
            self.stock_AdjClose = self.stock_price_history['Adj Close'][self.cur_step]
            self.stock_Volume = self.stock_price_history.Volume[self.cur_step]
            # indicators

            sma = calculateSMA(self.stock_price_history)

            if np.isnan(sma[self.cur_step]):
                self.sma = self.stock_close
            else:
                self.sma = sma[self.cur_step]

        ####################################################################

        ####################################################################
        ######################## state size of 11 ##########################
        elif self.state_size_used == 11:
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

        ####################################################################

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
        episodeDone = False
        # current value
        prev_val = self._get_val()

        ####################################################################
        ######################## state size of 3 ##########################
        if self.state_size_used == 3:
            self.stock_price = self.stock_price_history.Close[self.cur_step]

        ####################################################################

        ####################################################################
        ######################## state size of 4 ##########################
        elif self.state_size_used == 4:
            self.stock_open = self.stock_price_history.Open[self.cur_step]
            self.stock_close = self.stock_price_history.Close[self.cur_step]

        ####################################################################

        ####################################################################
        ######################## state size of 5 ##########################
        elif self.state_size_used == 5:
            self.stock_open = self.stock_price_history.Open[self.cur_step]
            self.stock_close = self.stock_price_history.Close[self.cur_step]
            self.stock_Volume = self.stock_price_history.Volume[self.cur_step]

        ####################################################################

        ####################################################################
        ######################## state size of 6 ##########################
        elif self.state_size_used == 6:
            self.stock_open = self.stock_price_history.Open[self.cur_step]
            self.stock_high = self.stock_price_history.High[self.cur_step]
            self.stock_low = self.stock_price_history.Low[self.cur_step]
            self.stock_close = self.stock_price_history.Close[self.cur_step]

        ####################################################################

        ####################################################################
        ######################## state size of 7 ##########################
        elif self.state_size_used == 7:
            self.stock_open = self.stock_price_history.Open[self.cur_step]
            self.stock_high = self.stock_price_history.High[self.cur_step]
            self.stock_low = self.stock_price_history.Low[self.cur_step]
            self.stock_close = self.stock_price_history.Close[self.cur_step]
            self.stock_Volume = self.stock_price_history.Volume[self.cur_step]

        ####################################################################

        ####################################################################
        ######################## state size of 8 ###########################
        elif self.state_size_used == 8:
            self.stock_open = self.stock_price_history.Open[self.cur_step]
            self.stock_high = self.stock_price_history.High[self.cur_step]
            self.stock_low = self.stock_price_history.Low[self.cur_step]
            self.stock_close = self.stock_price_history.Close[self.cur_step]
            self.stock_AdjClose = self.stock_price_history['Adj Close'][self.cur_step]
            self.stock_Volume = self.stock_price_history.Volume[self.cur_step]

        ####################################################################

        ####################################################################
        ######################## state size of 9 ###########################
        elif self.state_size_used == 9:
            self.stock_open = self.stock_price_history.Open[self.cur_step]
            self.stock_high = self.stock_price_history.High[self.cur_step]
            self.stock_low = self.stock_price_history.Low[self.cur_step]
            self.stock_close = self.stock_price_history.Close[self.cur_step]
            self.stock_AdjClose = self.stock_price_history['Adj Close'][self.cur_step]
            self.stock_Volume = self.stock_price_history.Volume[self.cur_step]
            # indicators

            sma = calculateSMA(self.stock_price_history)

            if np.isnan(sma[self.cur_step]):
                self.sma = self.stock_close
            else:
                self.sma = sma[self.cur_step]

        ####################################################################

        ####################################################################
        ######################## state size of 11 ##########################
        elif self.state_size_used == 11:
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

        ####################################################################

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
        obs[:self.n_stock] = self.stock_owned

        ####################################################################
        ######################## state size of 3 ##########################
        if self.state_size_used == 3:
            # [stockOwned, price, cashleft]

            obs[self.n_stock: 2 *self.n_stock] = self.stock_price
        ####################################################################
        elif self.state_size_used == 4:
            ####################################################################
            ######################## state size of 4 ##########################
            # [stockOwned, open, close, cashleft]
            obs[self.n_stock: 2 *self.n_stock] = self.stock_open
            obs[ 2 *self.n_stock: 3 *self.n_stock] = self.stock_close
        ####################################################################
        elif self.state_size_used == 5:
            ####################################################################
            ######################### state size of 5 ##########################
            # [stockOwned, open, close, volume, cashleft]
            obs[self.n_stock: 2 *self.n_stock] = self.stock_open
            obs[ 2 *self.n_stock: 3 *self.n_stock] = self.stock_close
            obs[ 3 *self.n_stock: 4 *self.n_stock] = self.stock_Volume
        ####################################################################
        elif self.state_size_used == 6:
            ####################################################################
            ######################### state size of 6 ##########################
            # [stockOwned, open, high, low, close, cashleft]
            obs[self.n_stock: 2 *self.n_stock] = self.stock_open
            obs[ 2 *self.n_stock: 3 *self.n_stock] = self.stock_high
            obs[ 3 *self.n_stock: 4 *self.n_stock] = self.stock_low
            obs[ 4 *self.n_stock: 5 *self.n_stock] = self.stock_close
        ####################################################################
        elif self.state_size_used == 7:
            ####################################################################
            ######################## state size of 7 ##########################
            # [stockOwned, open, high, low, close, Volume, cashleft]
            obs[self.n_stock: 2 *self.n_stock] = self.stock_open
            obs[ 2 *self.n_stock: 3 *self.n_stock] = self.stock_high
            obs[ 3 *self.n_stock: 4 *self.n_stock] = self.stock_low
            obs[ 4 *self.n_stock: 5 *self.n_stock] = self.stock_close
            obs[ 5 *self.n_stock: 6 *self.n_stock] = self.stock_Volume

        ####################################################################
        elif self.state_size_used == 8:
            ####################################################################
            ######################## state size of 8 ##########################
            # [stockOwned, open, high, low, close, adj close, Volume, cashleft]
            obs[self.n_stock: 2 *self.n_stock] = self.stock_open
            obs[ 2 *self.n_stock: 3 *self.n_stock] = self.stock_high
            obs[ 3 *self.n_stock: 4 *self.n_stock] = self.stock_low
            obs[ 4 *self.n_stock: 5 *self.n_stock] = self.stock_close
            obs[ 5 *self.n_stock: 6 *self.n_stock] = self.stock_AdjClose
            obs[ 6 *self.n_stock: 7 *self.n_stock] = self.stock_Volume
        ####################################################################
        elif self.state_size_used == 9:
            ####################################################################
            ######################## state size of 9 ##########################
            # [stockOwned, open, high, low, close, adj close, Volume, sma, cashleft]
            obs[self.n_stock: 2 *self.n_stock] = self.stock_open
            obs[ 2 *self.n_stock: 3 *self.n_stock] = self.stock_high
            obs[ 3 *self.n_stock: 4 *self.n_stock] = self.stock_low
            obs[ 4 *self.n_stock: 5 *self.n_stock] = self.stock_close
            obs[ 5 *self.n_stock: 6 *self.n_stock] = self.stock_AdjClose
            obs[ 6 *self.n_stock: 7 *self.n_stock] = self.stock_Volume
            obs[ 7 *self.n_stock: 8 *self.n_stock] = self.sma
        ####################################################################
        elif self.state_size_used == 11:
            ####################################################################
            ######################## state size of 11 ##########################
            # the current observation
            # [stockOwned, open, high, low, close, Adj Close, Volume,
            # rsi, sma, ema, cashleft]

            # [stockOwned,,,,,]
            # # [,open, high, low, close, Adj Close, Volume, rsi, sma, ema]
            obs[self.n_stock: 2 *self.n_stock] = self.stock_open
            obs[ 2 *self.n_stock: 3 *self.n_stock] = self.stock_high
            obs[ 3 *self.n_stock: 4 *self.n_stock] = self.stock_low
            obs[ 4 *self.n_stock: 5 *self.n_stock] = self.stock_close
            obs[ 5 *self.n_stock: 6 *self.n_stock] = self.stock_AdjClose
            obs[ 6 *self.n_stock: 7 *self.n_stock] = self.stock_Volume
            obs[ 7 *self.n_stock: 8 *self.n_stock] = self.rsi
            obs[ 8 *self.n_stock: 9 *self.n_stock] = self.sma
            obs[ 9 *self.n_stock:1 0 *self.n_stock] = self.ema

        ####################################################################
        ####################################################################
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
        if self.state_size_used < 4:
            ####################################################################
            ######################## state size of 3 ###########################
            return self.stock_owned * self.stock_price + self.cash_in_hand

        ####################################################################
        ###################### state size of 4 or more  #######################
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
            for i in sell_index:
                if self.state_size_used < 4:
                    ####################################################################
                    ######################### state size of 3 ##########################
                    self.cash_in_hand += self.stock_price * self.stock_owned
                ####################################################################
                ###################### state size of 4 or more #######################
                else:
                    self.cash_in_hand += self.stock_close * self.stock_owned
                    self.stock_owned = 0
        elif buy_index:
            can_buy = True
            while can_buy:
                for i in buy_index:
                    # buy a single stock on loop
                    # if there are multiple stocks then the loop iterates over each stock and buys 1 stock
                    # (if it can) for the stocks avaible each iteration.
                    if self.state_size_used < 4:
                        ####################################################################
                        ######################## state size of 3 ###########################
                        if self.cash_in_hand > self.stock_price:
                            self.stock_owned += 1  # buy one share
                            self.cash_in_hand -= self.stock_price
                        else:
                            # if the cash left is lower the stock price then stop buying
                            can_buy = False
                    ####################################################################
                    # ###################### state size of 4 or more #######################
                    else:
                        if self.cash_in_hand > self.stock_close:
                            self.stock_owned += 1  # buy one share
                            self.cash_in_hand -= self.stock_close
                        else:
                            # if the cash left is lower the stock price then stop buying
                            can_buy = False

        return allactions
