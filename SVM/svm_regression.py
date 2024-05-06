
import numpy as np
from sklearn.svm import SVR

from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from getData import get_historcial_data
from sklearn.model_selection import train_test_split


def fit_MinMax_scaller(X, y):
    # Normalize features and target variable
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_scaled = scaler.fit_transform(X)
    y_scaled = scaler.fit_transform(y.values.reshape(-1, 1))  # Reshape for 2D compatibility
    
    return X_scaled, y_scaled

def predict_prices(X_train, X_test, y_train, y_test):

    svr_lin = SVR(kernel='linear', C=1e3)
    svr_ploy = SVR(kernel='poly', C=1e3, degree=2)
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1) # You can experiment 
    # with different kernels

    svr_lin.fit(X_train, y_train.ravel())
    svr_ploy.fit(X_train, y_train.ravel())
    svr.fit(X_train, y_train.ravel())  # ravel() flattens the 2D array

    plt.scatter(X_train, y_train, color='black', lable='Data')
    plt.plot(X_train, svr_lin.predict(y_train), color='green', lable='Linear model')
    plt.plot(X_train, svr_ploy.predict(y_train), color='blue', lable='Ploynomial model')
    plt.plot(X_train, svr_rbf.predict(y_train), color='red', lable='RBF model')

    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('SVR')
    plt.show()

    return svr_lin.predict(y_train)[0], svr_ploy.predict(y_train)[0], svr_rbf.predict(y_train)[0]




if __name__ == "__main__":
    
    data = get_historcial_data("AAPL")
    
    y = data['Close']
    features = data[['Close']].shift(1)

    # Handle date indexing (using pandas.Timestamp.timestamp())
    timestamps = features.index.to_pydatetime()

    X = features.values  # Separate features from DataFrame
    X[0] = X[1]
    
    X_scaled, y_scaled = fit_MinMax_scaller(X, y)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2)

    pre_prices = predict_prices(X_train, X_test, y_train, y_test)
