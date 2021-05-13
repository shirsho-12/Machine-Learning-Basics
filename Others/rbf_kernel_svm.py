import numpy as np
import matplotlib.pyplot as plt
from iris_common_funcs import plot_decision_regions
from sklearn.svm import SVC
"""
Using the Radial Basis Function kernel (RBF Kernel)/ Gaussian Kernel to find and separate hyperplanes(decision
boundaries) in higher dimensions. i.e. to find decision boundaries of non-linear data

random dataset of 2 groups with 100 samples per group using logical xor
class labels: 1 & -1
"""
np.random.seed(0)
x_xor = np.random.randn(200, 2)
y_xor = np.logical_xor(x_xor[:, 0] > 0, x_xor[:, 1] > 0)
y_xor = np.where(y_xor, 1, -1)

"""
Plot of scatter data
plt.scatter(x_xor[y_xor == 1, 0], x_xor[y_xor == 1, 1], c='b', marker='x', label='1', edgecolor='black')
plt.scatter(x_xor[y_xor == -1, 0], x_xor[y_xor == -1, 1], c='r', marker='s', label='-1', edgecolor='black')
"""

# main function
svm = SVC(kernel='rbf', random_state=0, C= 10.0, gamma=0.10).fit(x_xor, y_xor)
plot_decision_regions(x_xor, y_xor, classifier=svm)

# plot
plt.ylim(-3.0)
plt.legend(loc='upper left')
plt.show()