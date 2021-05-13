"""
 Learning Curve using the WDBC Dataset to compare training and testing accuracies for different iterations of
 training data
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import learning_curve, validation_curve
from wdbc_common_funcs import wdbc_initializer, lr_pipeline


def plot(train_scores, test_scores, sizes, x_lab, scale=None):
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plt.plot(sizes, train_mean, color='blue', marker='o', markersize=5, label='training accuracy')
    plt.fill_between(sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
    plt.plot(sizes, test_mean, color='green', marker='s', markersize=5, linestyle='--', label='testing accuracy')
    plt.fill_between(sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')

    plt.grid()
    plt.xlabel(x_lab)
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.ylim([0.8, 1.0])
    if scale:
        plt.xscale(scale)
    plt.show()


x_train, x_test, y_train, y_test = wdbc_initializer()
pipe_lr = lr_pipeline()

train_sizes, train_scores, test_scores = learning_curve(estimator=pipe_lr, X=x_train, y=y_train,
                                                        train_sizes=np.linspace(0.1, 1.0, 10),
                                                        cv=10, n_jobs=1)

plot(train_scores, test_scores, train_sizes,'Number of training samples')

"""
 Verification Curve using the WDBC dataset to find training and testing accuracies for different values of gamma
 for Logistic Regression 
"""

params_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
train_scores, test_scores = validation_curve(estimator=pipe_lr, X=x_train, y=y_train, param_name="lr__C",
                                             param_range=params_range, cv=10)

plot(train_scores, test_scores, params_range, 'Parameter C', scale='log')


