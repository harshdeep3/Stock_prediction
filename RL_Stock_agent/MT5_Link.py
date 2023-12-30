import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime
import pytz

LOGIN = 51535169
SERVER = "ICMarketsSC-Demo"
# password
PASSWORD = "Cdm9I@7hsU"

# display data on the MetaTrader 5 package
print("MetaTrader5 package author: ", mt5.__author__)
print("MetaTrader5 package version: ", mt5.__version__)


class MT5Class:

    def __init__(self):
        self.mt5_result = None
        self.account_info = None

    def login_to_metatrader(self):
        # Connect to the MetaTrader 5 terminal
        mt5.initialize()

        # Log in to the terminal with your account credentials
        account_server = SERVER
        # this needs to be an integer
        login = LOGIN
        password = PASSWORD
        self.mt5_result = mt5.login(login, password, account_server)

        if not self.mt5_result:
            print("Login failed. Check your credentials.")
            quit()

    def get_acc_info(self):

        if mt5.account_info() is None:
            print("Account info is None!")
        else:
            account_info_dict = mt5.account_info()._asdict()
            self.account_info = pd.DataFrame(list(account_info_dict.items()), columns=['property', 'value'])
            print(self.account_info)


def get_historic_data(fx_symbol, fx_timeframe, fx_count):

    rates = mt5.copy_rates_from_pos(fx_symbol, fx_timeframe, 0, fx_count)
    # dataframe
    historic_df = pd.DataFrame(rates)
    # changing the time to datetime
    if "time" in historic_df.keys():
        historic_df['time'] = pd.to_datetime(historic_df['time'], unit='s')
        return historic_df
    else:
        print("\n\nData not found! Check MT5 connection!")
        return None


if __name__ == "__main__":
    mt5_obj = MT5Class()
    mt5_obj.login_to_metatrader()
    mt5_obj.get_acc_info()

    # timeframe objects https://www.mql5.com/en/docs/python_metatrader5/mt5copyratesfrom_py
    timeframe = mt5.TIMEFRAME_D1
    symbol = 'USDJPY'
    # set time zone to UTC
    timezone = pytz.timezone("Etc/UTC")
    # create 'datetime' objects in UTC time zone to avoid the implementation of a local time zone offset
    utc_from = datetime(2010, 1, 10, tzinfo=timezone)
    utc_to = datetime(2020, 1, 11, tzinfo=timezone)
    
    # goes back to 1971-08-11
    count = 13500

    # print account info
    # mt5_obj.get_acc_info()
    
    # get data
    df = get_historic_data(symbol, timeframe, count)
    df = df.set_index('time')

    print(df)
    # Disconnect from the terminal
    mt5.shutdown()
