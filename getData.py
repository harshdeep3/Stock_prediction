
import numpy as np

import yfinance as yf

def get_historcial_data(ticker_name):
  return yf.download(ticker_name)


if __name__ == "__main__":
    
    data = print(get_historcial_data("AAPL"))