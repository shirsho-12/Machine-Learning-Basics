"""
Using the RBF Kernel PCA on Sci-Kit Learn's make_circles dataset
"""

from sklearn.datasets import make_circles
from sklearn.decomposition import PCA
from rbf_kernel_pca import rbf_kernel_pca
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter


def plot(x_pca, y):
    ax[0].scatter(x_pca[y == 0, 0], x_pca[y == 0, 1], c='red', marker='^', alpha=0.5)
    ax[0].scatter(x_pca[y == 1, 0], x_pca[y == 1, 1], c='blue', marker='o', alpha=0.5)
    ax[1].scatter(x_pca[y == 0, 0], np.zeros((500, 1)) + 0.02, c='red', marker='^', alpha=0.5)
    ax[1].scatter(x_pca[y == 1, 0], np.zeros((500, 1)) - 0.02, c='blue', marker='o', alpha=0.5)

    ax[0].set_xlabel('PC1')
    ax[0].set_ylabel('PC2')
    ax[1].set_ylim([-1, 1])
    ax[1].set_yticks([])
    ax[1].set_xlabel('PC1')

    ax[0].xaxis.set_major_formatter(FormatStrFormatter('%0.1f'))
    ax[1].xaxis.set_major_formatter(FormatStrFormatter('%0.1f'))
    plt.show()


x, y = make_circles(n_samples=1000, random_state=123, noise=0.1, factor=0.2)

"""
plt.scatter(x[y == 0, 0], x[y == 0, 1], c='red', marker='^', alpha=0.5)
plt.scatter(x[y == 1, 0], x[y == 1, 1], c='blue', marker='o', alpha=0.5)
plt.show()
"""

scikit_pca = PCA(n_components=2)
x_sc_pca = scikit_pca.fit_transform(x)
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(7, 3))
# plot(x_sc_pca, y)

x_k_pca = rbf_kernel_pca(x, gamma=5, n_components=2)
plot(x_k_pca, y)

