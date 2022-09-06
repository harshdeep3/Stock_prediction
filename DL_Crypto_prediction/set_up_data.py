import numpy as np
import pandas as pd
import pandas_datareader as web

from datetime import datetime
from sklearn.preprocessing import MinMaxScaler


DATE_START = datetime(2010, 7, 1)
DATE_END = datetime.today()


def get_data(stock_name, start, end):
    """
    This gets the data which is used by the agent.
    Returns:
    dataframe [dataframe]: This returns the dataframe which is used for
    the state and by the agent
    :param stock_name:
    :param start:
    :param end:
    :return:
    """
    df = web.DataReader(stock_name, 'yahoo', start, end)
    # reset the index of the data to normal ints so df['Date'] can be used
    df.reset_index()
    return df


def get_scaled_data(data, scaler=None):
    if scaler is None:
        scaler = MinMaxScaler(feature_range=(0, 1))

    scaled_data = scaler.fit_transform(data.reshape(-1, 1))
    # plot scaled data

    # pd.DataFrame(scaled_data).plot()

    return scaled_data, scaler


def create_pattern_set(data_target, steps=7):
    x_train = []
    y_train = []

    for day in range(steps, data_target.shape[0]):
        x_train.append(data_target[day-steps:day])
        y_train.append(data_target[day])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 2))

    return x_train, y_train


def prepare_data(df, step=50):
    train_df_len = int(len(df) * 0.8)
    test_df_len = len(df) - train_df_len

    whole_train_data = df.iloc[:train_df_len]
    whole_test_data = df[train_df_len - step:]

    # first dim -> close data
    close_train_data = whole_train_data['Close'].values.reshape(train_df_len, 1)
    close_test_data = whole_test_data['Close'].values.reshape(-1, 1)

    # 2nd dim -> Volume
    vol_train_data = whole_train_data['Volume'].values.reshape(train_df_len, 1)
    vol_test_data = whole_test_data['Volume'].values.reshape(-1, 1)

    # scaled data
    scaled_close_train_data, close_scaler = get_scaled_data(close_train_data)
    scaled_close_test_data = close_scaler.transform(close_test_data)

    scaled_vol_train_data, vol_scaler = get_scaled_data(vol_train_data)
    scaled_vol_test_data = vol_scaler.transform(vol_test_data)

    # combine data
    combined_train_data = np.concatenate((scaled_close_train_data, scaled_vol_train_data), axis=1)
    combined_test_data = np.concatenate((scaled_close_test_data, scaled_vol_test_data), axis=1)

    x_train, y_train = create_pattern_set(combined_train_data, steps=50)
    x_test, y_test = create_pattern_set(combined_test_data, steps=50)

    y_train = np.reshape(y_train, (y_train.shape[0], 2))
    y_test = np.reshape(y_test, (y_test.shape[0], 2))

    return x_train, x_test, y_train, y_test, close_scaler
