# Creating a perceptron using scikit_learn's Perceptron class and the iris dataset

import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron
from iris_common_funcs import plot_decision_regions, initializer


X_train_std, y_train, X_combined_std, y_combined = initializer()
ppn = Perceptron(max_iter=40, eta0=0.1, random_state=0).fit(X_train_std, y_train)
plot_decision_regions(X=X_combined_std, y=y_combined, classifier=ppn, test_idx=range(105, 150))

plt.xlabel("Sepal length (standardized)")
plt.ylabel("Petal length (standardized)")
plt.legend(loc='upper left')
plt.show()