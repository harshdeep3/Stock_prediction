import numpy as np
import torch
from torch import optim


class LinearModel:

    def __init__(self, input_dim, n_action):
        # tensor create with a set size using normal distribution
        self.W = torch.randn(
            input_dim, n_action) / np.sqrt(input_dim)
        self.b = torch.zeros(n_action, dtype=torch.float64, requires_grad=True)

        self.W = self.W.type(torch.DoubleTensor)
        self.W.requires_grad_(True)
        # momentum terms
        self.vW = 0
        self.vb = 0

        self.losses = []

    def predict(self, X):
        if type(X) == np.ndarray:
            X = torch.from_numpy(X)
            X.requires_grad_(True)

        ans = X.double() @ self.W.double() + self.b.double()
        return ans

    def sgd(self, X, Y, learning_rate=0.001, momentum=0.5):
        # this changes the type of X and Y to torch tensors
        if type(self.W) == np.ndarray:
            self.W = torch.from_numpy(self.W)
            self.W.requires_grad_(True)

        if type(self.b) == np.ndarray:
            self.b = torch.from_numpy(self.b)
            self.b.requires_grad_(True)

        # to get the gradients
        X = torch.from_numpy(X)
        X.requires_grad_(True)
        Y = torch.from_numpy(Y)
        Y.requires_grad_(True)
        # inbuilt SGD function from torch
        optimizer = optim.SGD([self.b, self.W], lr=learning_rate, momentum=momentum)

        yHat = self.predict(X)

        # inbuilt function for mean squared error
        loss = torch.nn.MSELoss()
        mse = loss(yHat, Y)
        # gets the gradients
        mse.backward()
        with torch.no_grad():
            # this updates the weights and bias
            optimizer.step()
            optimizer.zero_grad()
        # changes to a numpy
        mse = mse.detach().numpy()
        self.losses.append(np.sqrt(mse))

    def load_weights(self, filepath):
        """[summary]
        Load the saved weight and bias after training them
        Args:
                filepath ([type]): [description] Location where there are save
        """
        w = torch.load(filepath)

        self.W = w['W']
        self.b = w['b']

    def save_weights(self, filepath):
        """[summary]
        Save the weigth and bias after training is complete so they can be used later
        Args:
                filepath ([type]): [description] Location where the file is saved.
        """

        torch.save({"W": self.W, "b": self.b}, filepath)
