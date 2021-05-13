"""
SVM - Support Vector Machine
Uses points closest to the decision boundary to yield results
Less prone to outliers
Similar results to Logistic Regression but a comparatively more complex model
"""

from sklearn.svm import SVC
from iris_common_funcs import plot_decision_regions, initializer
import matplotlib.pyplot as plt

X_train_std, y_train, X_combined_std, y_combined = initializer()


def linear_plot():
    svm = SVC(kernel='linear', C=1.0, random_state=0).fit(X_train_std, y_train)
    plot_decision_regions(X=X_combined_std, y=y_combined, classifier=svm, test_idx=range(105, 150))


def non_linear_plot():
    svm = SVC(kernel='rbf', random_state=0, gamma=0.2, C=1.0).fit(X_train_std, y_train)
    plot_decision_regions(X=X_combined_std, y=y_combined, classifier=svm, test_idx=range(105, 150))


non_linear_plot()
plt.xlabel("Sepal length (standardized)")
plt.ylabel("Petal length (standardized)")
plt.legend(loc='upper left')
plt.show()
