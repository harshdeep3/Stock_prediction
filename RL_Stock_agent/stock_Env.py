from stockEnv import StockMarketEnv as GenEnv
import MT5_Link as link
import MetaTrader5 as mt5


class Env(GenEnv):
    """

    This is a wrapper around the Environment, this allows the stable baseline3 to create multiple environment.

    """

    def __init__(self, timeframe=mt5.TIMEFRAME_D1, symbol='USDJPY', count=13500):

        # get historic data 
        data = link.get_historic_data(fx_symbol=symbol, fx_timeframe=timeframe, fx_count=count)
        # set time as index
        data = data.set_index('time')
        
        if data is None:
            print("Error: Data not recieved!")
        else:
            super(Env, self).__init__(data)

