import numpy as np
import pandas as pd
import tensorflow as tf
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import math, time

from sklearn.metrics import mean_squared_error
from set_up_data import get_data, prepare_data, DATE_START, DATE_END


class LSTM(nn.Module):

    def __init__(self, input_dim=1, hidden_dim=32, num_layer=2, output_dim=1):
        super(LSTM, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layer

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layer, batch_first=True)

        # readout layer -> flatten
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):

        # initialise hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()

        # initialise cell state
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()

        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        # Index hidden state of last time step
        # print(out.size())
        # out.size() --> 100, 32, 100
        # out[:, -1, :] --> 100, 100 --> just want last time step hidden states!
        out = self.fc(out[:, -1, :])
        # print(out.size())
        # out.size() --> 100, 10
        return out


def train(model, loss_fn, optimiser, x_train, y_train, step):
    num_epochs = 100
    hist = np.zeros(num_epochs)

    # num of steps to unroll
    seq_dim = step - 1

    for epoch in range(num_epochs):
        # Initialise hidden state
        # Don't do this if you want your LSTM to be stateful
        # model.hidden = model.init_hidden()

        y_train_pred = model(x_train)
        loss = loss_fn(y_train_pred, y_train)

        if epoch % 10 == 0 and epoch != 0:
            print("Epoch ", epoch, " MSE: ", loss.item())

        hist[epoch] = loss.item()

        # zero out gradient, else they will accumulate between epochs
        optimiser.zero_grad()

        # backward pass
        loss.backward()

        # update parameters
        optimiser.step()

    return hist, y_train_pred


def main():
    step = 7

    df = get_data('BTC-USD', DATE_START, DATE_END)
    x_train, x_test, y_train, y_test, scaler = prepare_data(df, step)

    # convert to torch tensor
    x_train = torch.from_numpy(x_train).type(torch.Tensor)
    x_test = torch.from_numpy(x_test).type(torch.Tensor)
    y_train = torch.from_numpy(y_train).type(torch.Tensor)
    y_test = torch.from_numpy(y_test).type(torch.Tensor)

    # check data
    # plt.plot(df['High'])
    # plt.show()
    # print("x_train ", x_train, "\nx_test ", x_test, "\ny_train", y_train, "\ny_test", y_test)

    # print("x train ", x_train.size(), "x test ", x_test.size())
    # print("y train ", y_train.size(), "y test ", y_test.size())

    model = LSTM()
    loss_fn = torch.nn.MSELoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=0.01)

    # show model structure
    # print(model)
    # print(len(list(model.parameters())))
    # for i in range(len(list(model.parameters()))):
    #     print(list(model.parameters())[i].size())

    hist, y_train_pred = train(model, loss_fn, optimiser, x_train, y_train, step)

    # plot the loss for each epoch
    plt.plot(hist, label="Training Loss")
    plt.show()

    # make predictions
    y_test_pred = model(x_test)

    # invert predictions
    y_train_pred = scaler.inverse_transform(y_train_pred.detach().numpy())
    y_train = scaler.inverse_transform(y_train.detach().numpy())
    y_test_pred = scaler.inverse_transform(y_test_pred.detach().numpy())
    y_test = scaler.inverse_transform(y_test.detach().numpy())

    # calculate root mean squared error
    train_score = math.sqrt(mean_squared_error(y_train[:, 0], y_train_pred[:, 0]))
    print('Train Score: %.2f RMSE' % train_score)
    test_score = math.sqrt(mean_squared_error(y_test[:, 0], y_test_pred[:, 0]))
    print('Test Score: %.2f RMSE' % test_score)

    # Visualising the results
    figure, axes = plt.subplots(figsize=(15, 6))
    axes.xaxis_date()

    axes.plot(df[len(df) - len(y_test):].index, y_test, color='red', label='Real BTC Stock Price')
    axes.plot(df[len(df) - len(y_test):].index, y_test_pred, color='blue', label='Predicted BTC Stock Price')
    # axes.xticks(np.arange(0,394,50))
    plt.title('BTC Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('BTC Stock Price')
    plt.legend()
    plt.savefig('BTC_pred.png')
    plt.show()


if __name__ == '__main__':
    main()
