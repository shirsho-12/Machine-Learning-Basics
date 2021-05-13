"""
Using a random forest of a 1000 trees to rank the 13 features of the wine dataset
"""

from sklearn.ensemble import RandomForestClassifier
from wine_comon_funcs import wine_initializer
import numpy as np
import matplotlib.pyplot as plt

x_train, y_train, x_test, y_test, columns = wine_initializer('val')
feat_labels = columns[1:]

forest = RandomForestClassifier(n_estimators=1000, random_state=0, n_jobs=-1).fit(x_train, y_train)
# n_jobs = -1 means using all processors

importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]

for f in range(x_train.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30, feat_labels[f], importances[indices[f]]))
    # print("{0}) {1}{2:<30.5f}".format(f + 1, feat_labels[f], importances[indices[f]]))

plt.title("Feature Importances")
plt.bar(range(x_train.shape[1]), importances[indices], color='lightblue', align='center')
plt.xticks(range(x_train.shape[1]), feat_labels, rotation=90)
plt.xlim([-1, x_train.shape[1]])
plt.tight_layout()
plt.show()
