"""
Training a simple Linear Regression model using OLS (ordinary least squares) with the Theano framework.
"""

import theano
import numpy as np
import matplotlib.pyplot as plt
from theano import tensor as T

theano.config.floatX='float32'
d_type = theano.config.floatX


def train_lin_reg(x_train, y_train, l_rate, epochs):
    costs = []
    # initialize arrays
    l_rate0 = T.fscalar('l_rate0')
    y = T.fvector(name='y')
    x = T.fmatrix(name='x')
    w = theano.shared(np.zeros(shape=x_train.shape[1] + 1, dtype=d_type))

    # calculate cost
    net_input = T.dot(x, w[1:]) + w[0]                 # w[0] is the bias unit (y-intercept)
    errors = y - net_input
    cost = T.sum(T.pow(errors, 2))                     # OLS squaring

    # perform gradient update
    gradient = T.grad(cost, wrt=w)                      # derivative (gradient) computation
    update = [(w, w - l_rate0 * gradient)]              # update for compilation

    # compile model
    train = theano.function(inputs=[l_rate0], outputs=cost,
                            updates=update, givens={x: x_train, y: y_train})
    for _ in range(epochs):
        costs.append(train(l_rate))

    return costs, w


def predict_lin_reg(x, w):
    x_t = T.matrix(name='x')
    net_input = T.dot(x_t, w[1:]) + w[0]
    predict = theano.function(inputs=[x_t], givens={w: w}, outputs=net_input)

    return predict(x)


x_train = np.asarray([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0], [6.0],
                      [7.0], [8.0], [9.0]], dtype=d_type)
y_train = np.asarray([1.0, 1.3, 3.1, 2.0, 5.0, 6.3, 6.6, 7.4, 8.0, 9.0], dtype=d_type)

costs, w = train_lin_reg(x_train, y_train, l_rate=0.001, epochs=10)
plt.plot(range(1, 1 + len(costs)), costs)
plt.tight_layout()
plt.xlabel("Epoch")
plt.ylabel('Cost')
plt.show()

plt.scatter(x_train, y_train, marker='s', s=50, edgecolors='black')
plt.plot(range(x_train.shape[0]), predict_lin_reg(x_train, w), color='gray',
         marker='o', markersize=4, linewidth=3)
plt.xlabel('x')
plt.ylabel('y')
plt.show()
