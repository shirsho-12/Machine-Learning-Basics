# Logistic regression model using scikit learn

from sklearn.linear_model import LogisticRegression
from iris_common_funcs import plot_decision_regions, initializer
import matplotlib.pyplot as plt
import numpy as np


X_train_std, y_train, X_combined_std, y_combined = initializer()

lr = LogisticRegression(C=1000.0, random_state=0).fit(X_train_std, y_train)  # single plot
plot_decision_regions(X=X_combined_std, y=y_combined, classifier=lr, test_idx=range(105, 150))

plt.xlabel("Sepal length (standardized)")
plt.ylabel("Petal length (standardized)")
plt.legend(loc='upper left')
plt.show()


def regularization_parameter_plot():
    weights, params = [],[]
    for c in np.arange(-5, 5):  # multiple plots
        l_r = LogisticRegression(C=10.0 ** c, random_state=0).fit(X_train_std, y_train)
        weights.append(l_r.coef_[1])          # coef_ = 1/ C
        params.append(10.0 ** c)

    weights = np.array(weights)
    plt.plot(params, weights[:, 0], label='petal-length')
    plt.plot(params, weights[:, 1], linestyle='--', label='petal-width')
    plt.ylabel('weight coefficient')
    plt.xlabel('C')
    plt.legend(loc='upper left')
    plt.xscale('log')
    plt.show()
