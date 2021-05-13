"""
Function to calculate error rates in ensemble learning- multiple learners are trained to solve one problem
and plot of ensemble error vs base error
"""
from scipy.special import comb
import math
import numpy as np
import matplotlib.pyplot as plt


def ensemble_error(n_classifier, error):
    k_start = math.ceil(n_classifier / 2.0)
    probabilities = [comb(n_classifier, k) * error ** k * (1 - error) ** (n_classifier - k) for
                     k in range(k_start, n_classifier + 1)]

    return sum(probabilities)


# print(ensemble_error(n_classifier=11, error=0.25))
error_range = np.arange(0.0, 1.01, 0.01)
ens_errors = [ensemble_error(n_classifier=11, error=error) for error in error_range]
plt.plot(error_range, ens_errors, label='Ensemble Error', linewidth=2)
plt.plot(error_range, error_range, linestyle='--', label='Base Error', linewidth=2)
plt.xlabel('Base Error')
plt.ylabel("Ensemble Error")
plt.grid()
plt.show()