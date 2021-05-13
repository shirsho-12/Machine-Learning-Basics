"""
Using the iris data set and the KNN(k-nearest neighbor lazy learning algorithm) to perform
SBS(sequential backward selection).
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sbs import SBS
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from wine_comon_funcs import wine_initializer


x_train_std, y_train, x_test_std, y_test, __ = wine_initializer()

knn = KNeighborsClassifier(n_neighbors=2)
sbs = SBS(knn, k_features=1)
sbs.fit(x_train_std, y_train)

k_feat = list(len(k) for k in sbs.subsets_)
plt.plot(k_feat, sbs.scores_, marker='o')
plt.ylim([0.7, 1.1])
plt.ylabel("Accuracy")
plt.xlabel("Number of features")
plt.grid()
plt.show()

k_5 = list(sbs.subsets_[8])
knn.fit(x_train_std[:, k_5], y_train)   # Higher accuracy of test data with less overfitting using less dimensions
# print(df_wine.columns[1:][k_5])
print('Training accuracy:', knn.score(x_train_std[:, k_5], y_train))
print('Test accuracy:', knn.score(x_test_std[:, k_5], y_test))


