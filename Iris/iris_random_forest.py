"""
Ensemble learning - Random Forests
Decision boundaries created using multiple decision trees combined together
"""

from sklearn.ensemble import RandomForestClassifier
from iris_common_funcs import plot_decision_regions, initializer
import matplotlib.pyplot as plt

x_train, y_train, x_combined, y_combined = initializer('val')
forest = RandomForestClassifier(criterion='entropy', n_estimators=10, random_state=1, n_jobs=2)
forest.fit(x_train, y_train)
plot_decision_regions(x_combined, y_combined, classifier=forest, test_idx=range(105,150))

plt.xlabel("Sepal length")
plt.ylabel("Petal length")
plt.legend(loc='upper left')
plt.tick_params(top=True, right=True)   # adds ticks to the sides
plt.show()

