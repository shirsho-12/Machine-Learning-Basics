"""
Comparision between 3 activation functions: logistic, softmax and tanh
"""

import numpy as np
import matplotlib.pyplot as plt


def net_input(x, w):
    z = x.dot(w)
    return z


# Logistic function: l(z) = 1 / (1 + e^-z)
# expit(z)  (from scipy.special import expit)
def logistic(z):
    return 1.0 / (1.0 + np.exp(-z))


def logistic_activation(x, w):
    z = net_input(x, w)
    return logistic(z)


x = np.array([[1, 1.4, 1.5]])
w = np.array([0.0, 0.2, 0.4])
print('P(y=1|x) = {0:.3f}'.format(logistic_activation(x, w)[0]))


"""
An output layer with multiple logistic activation layers (i.e. hidden layers) does not produce
meaningful, interpretable probability values
"""
# w : array, shape = [n_output_units, n_hidden_units+1]. Weight matrix for hidden layer -> output layer.
# The first column (A[:][0] = 1) are the bias units

w = np.array([[1.1, 1.2, 1.3, 0.5], [0.1, 0.2, 0.4, 0.1], [0.2, 0.5, 2.1, 1.9]])

# a : array, shape = [n_hidden+1, n_samples]. Activation of hidden layer.
# The that first element (A[0][0] = 1) is the bias unit
a = np.array([[1.0], [0.1], [0.3], [0.7]])

# z : array, shape = [n_output_units, n_samples] Net input of the output layer.
z = w.dot(a)
y_probas = logistic(z)
print("Probabilities: \n", y_probas)

"""Does not matter if the model is used to predict class labels, not class membership probabilities"""
y_class = np.argmax(z, axis=0)
print("Predicted class label: {0}".format(y_class[0]))

"""
Softmax: generalized logistic function with  a normalization term(sum of all logistic 
outputs on an array).
Allows computation of meaningful class probabilities in multi=class settings (as opposed to OvR settings)
"""


def softmax(z):
    return np.exp(z) / np.sum(np.exp(z))


def sofmax_activation(x, w):
    z = net_input(x, w)
    return softmax(z)


y_probas = softmax(z)
print("Probabilities: \n", y_probas)
"""Same predicted class label output(2)"""

"""
tanh: hyperbolic tangent = 2 * logistic(2z) - 1
Double the output range compared to the logistic function (-1 -> 1 instead of 0 -> 1)
"""


# np.tanh(z)
def tanh(z):
    e_p = np.exp(z)
    e_m = np.exp(-z)
    return (e_p - e_m) / (e_p + e_m)


z = np.arange(-5, 5, 0.005)
log_act = logistic(z)
tanh_act = tanh(z)

plt.ylim([-1.5, 1.5])
plt.xlabel('net input $z$')
plt.ylabel('activation $\phi(z)$')
for i in zip([-1, 0, 0.5, 1]):
    plt.axhline(i, color='black', linestyle='--')

plt.plot(z, log_act, color='lightgreen', label='Logistic')
plt.plot(z, tanh_act, color='black', label='tanh')

plt.legend(loc='best')
plt.tight_layout()
plt.show()
