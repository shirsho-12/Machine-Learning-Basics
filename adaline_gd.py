from perceptron import Perceptron
import numpy as np
from numpy.random import seed


class AdalineGD(Perceptron):
    """ADAptive LInear NEuron classifier.

    Attributes(new for stochastic gradient descent):
        shuffle - bool(default:True)
                  Shuffles training data after every epoch. If True to prevent cycles.
        random_state - int(default:None)
                       Set random state for shuffling and initialising the weights.
    """

    def __init__(self, l_rate, n_iter, shuffle=True, random_state = None):
        super().__init__(l_rate, n_iter)
        self.w_initialised = False   # weight array initialised
        self.shuffle = shuffle

        if random_state:
            seed(random_state)
    """
    Batch Gradient Descent fit function
    def fit(self, X, y):            
        self.w_arr_ = np.zeros(1 + X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            output = self.net_input(X)
            errors = (y - output)
            self.w_arr_[1:] += self.l_rate * X.T.dot(errors)
            self.w_arr_[0] += self.l_rate * errors.sum()
            cost = (errors ** 2).sum() / 2.0
            self.cost_.append(cost)
        return self
    """

    def fit(self, X, y):
        self._initialise_weights(X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            if self.shuffle:
                X, y = self._shuffle(X, y)
            cost = []
            for x_i, target in zip(X, y):
                cost.append(self._update_weights(x_i, target))
            avg_cost = sum(cost) / len(y)
            self.cost_.append((avg_cost))

        return self
    """New functions for stochastic gradient descent"""
    def partial_fit(self, X, y):
        """ Fit training without initialising the weights"""
        if not self.w_initialised:
            self._initialise_weights(X.shape[1])
        if y.ravel().shape[0] > 1:
            for x_i, target in zip(X, y):
                self._update_weights(x_i, target)
        else:
            self._update_weights(X, y)
        return self

    def _shuffle(self, X, y):
        r = np.random.permutation(len(y))
        return X[r], y[r]

    def _initialise_weights(self, m):
        self.w_arr_ = np.zeros(1 + m)
        self.w_initialised = True

    def _update_weights(self, x_i, target):
        """Apply Adaline learning rule to update the weights"""
        output = self.net_input(x_i)
        error = target - output
        self.w_arr_[1:] += self.l_rate * x_i.dot(error)
        self.w_arr_[0] += self.l_rate * error
        cost = 0.5 * error ** 2
        return cost

    def activation(self, X):
        return self.net_input(X)
        
    
            