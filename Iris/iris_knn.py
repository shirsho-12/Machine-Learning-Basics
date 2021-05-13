"""
KNN - k-nearest neighbor classifier
Lazy learning algorithm
Quick adaptation to new test data
Linear increase in computational complexity
"""
from sklearn.neighbors import KNeighborsClassifier
from iris_common_funcs import plot_decision_regions, initializer
import matplotlib.pyplot as plt

X_train_std, y_train, X_combined_std, y_combined = initializer()
knn= KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski').fit(X_train_std, y_train)
'''
minkowski distance metric is a generalisation of the Euclidean and Manhattan distance metrics
At p=2, Euclidean distance is used and at p=1, the distance is Manhattan.
'''
plot_decision_regions(X=X_combined_std, y=y_combined, classifier=knn, test_idx=range(105, 150))

plt.xlabel("Sepal length (standardized)")
plt.ylabel("Petal length (standardized)")
plt.legend(loc='upper left')
plt.tick_params(top=True, right=True) 
plt.show()
