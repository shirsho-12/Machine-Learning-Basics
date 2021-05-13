"""
Decision Tree built using skikit learn and the iris dataset
"""

from sklearn.tree import DecisionTreeClassifier, export_graphviz
from iris_common_funcs import plot_decision_regions, initializer
import matplotlib.pyplot as plt

x_train, y_train, x_combined, y_combined = initializer('val')
tree = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0).fit(x_train, y_train)
plot_decision_regions(x_combined, y_combined, classifier=tree, test_idx=range(105,150))

plt.xlabel("Sepal length [cm]")
plt.ylabel("Petal length [cm]")
plt.legend(loc='upper left')
plt.show()

export_graphviz(tree, out_file='iris_tree.dot', feature_names=['petal length', 'petal width'])