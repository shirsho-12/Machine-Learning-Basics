# Algorithm using Rosenblatt's perception rule
import numpy as np


class Perceptron():
    """Perceptron classifier
    Parameters:
          l_rate - learning rate (between 0.0 and 1.0)
          n_iter - passes over the training dataset, i.e epochs

    Attributes:
          w_arr_ - 1d array of weights after fitting
          errors_ - list of misclassifications in every epoch (an epoch is a pass over a training dataset)
    """
    def __init__(self, l_rate = 0.01, n_iter = 10):
        self.l_rate = l_rate
        self.n_iter = n_iter

    def fit(self, X, y):
        """Fit training data into an array
        Parameters:
            X : {array-like} shape - [n_samples, n_features]     Training vectors
            y : {array-like} shape - [n_samples]                 Target values

        Returns:
              self: object
        """
        self.w_arr_ = np.zeros(1 + X.shape[1])
        self.errors_ = []

        for i in range(self.n_iter):
            errors = 0
            for x_i, target in zip(X, y):
                update = self.l_rate * (target - self.predict(x_i))
                self.w_arr_[1:] += update * x_i
                self.w_arr_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):         # calculate net input
        return np.dot(X, self.w_arr_[1:]) + self.w_arr_[0]

    def predict(self, X):           # Returns class label after unit step
        return np.where(self.net_input(X) >= 0.0, 1, -1)
