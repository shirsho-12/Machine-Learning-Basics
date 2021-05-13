import matplotlib.pyplot as plt
from sklearn import datasets
from matplotlib.colors import ListedColormap
import numpy as np

iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target
markers = ('s', 'x', 'o', '^', 'v')
colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
cmap = ListedColormap(colors[:len(np.unique(y))])

for index, cl in enumerate(np.unique(y)):
    plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8,
                c=cmap(index), marker=markers[index], label=cl, edgecolor='black')
plt.ylabel('y-axis')
plt.xlabel('x-axis')
plt.show()

