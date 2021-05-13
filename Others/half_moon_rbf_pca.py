"""
Using the RBF Kernel PCA on Sci-Kit Learn's half-moon dataset
"""

from sklearn.datasets import make_moons
from sklearn.decomposition import PCA
from rbf_kernel_pca import rbf_kernel_pca
import matplotlib.pyplot as plt
import numpy as np

"""
def plot(x_pca, y):
    ax[0].scatter(x_pca[y == 0, 0], x_pca[y == 0, 1], c='red', marker='^', alpha=0.5)
    ax[0].scatter(x_pca[y == 1, 0], x_pca[y == 1, 1], c='blue', marker='o', alpha=0.5)
    ax[1].scatter(x_pca[y == 0, 0], np.zeros((50, 1)) + 0.02, c='red', marker='^', alpha=0.5)
    ax[1].scatter(x_pca[y == 1, 0], np.zeros((50, 1)) - 0.02, c='blue', marker='o', alpha=0.5)

    ax[0].set_xlabel('PC1')
    ax[0].set_ylabel('PC2')
    ax[1].set_ylim([-1, 1])
    ax[1].set_yticks([])
    ax[1].set_xlabel('PC1')

    ax[0].xaxis.set_major_formatter(FormatStrFormatter('%0.1f'))
    ax[1].xaxis.set_major_formatter(FormatStrFormatter('%0.1f'))
    plt.show()
"""

def project_x(x_new, x, gamma, alphas, lambdas):
    pair_dist = np.array([np.sum((x_new - row) ** 2) for row in x])
    k = np.exp(-gamma * pair_dist)
    return k.dot(alphas/lambdas)

x, y = make_moons(n_samples=100, random_state=123)

"""
plt.scatter(x[y == 0, 0], x[y == 0, 1], c='red', marker='^', alpha=0.5)
plt.scatter(x[y == 1, 0], x[y == 1, 1], c='blue', marker='o', alpha=0.5)
plt.show()
"""

scikit_pca = PCA(n_components=2)
x_sc_pca = scikit_pca.fit_transform(x)
# fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(7, 3))
# plot(x_sc_pca, y)

alphas, lambdas = rbf_kernel_pca(x, gamma=15, n_components=1)
x_new, x_proj = x[25], alphas[25]
# print (x_new, x_proj)

x_reproj = project_x(x_new, x, gamma=15, alphas=alphas, lambdas=lambdas)
# print(x_reproj)
plt.scatter(alphas[y == 0, 0], np.zeros((50)), c='red', marker='^', alpha=0.5)
plt.scatter(alphas[y == 1, 0], np.zeros((50)), c='blue', marker='o', alpha=0.5)
plt.scatter(x_proj, 0, color='black', label='original projection of point X[25]', marker='^', s=100)
plt.scatter(x_reproj, 0, color='green', label='remapped point X[25]', marker='x', s=500)
plt.legend(scatterpoints=1)
plt.show()
# plot(x_k_pca, y)


"""Using scikit-learn's kernel PCA to produce similar results"""

from sklearn.decomposition import KernelPCA

scikit_kpca = KernelPCA(n_components=2, kernel='rbf', gamma=15)
x_sk_kern_pca = scikit_kpca.fit_transform(x)
plt.scatter(x_sk_kern_pca[y == 0, 0], x_sk_kern_pca[y == 0, 1], c='red', marker='^', alpha=0.5)
plt.scatter(x_sk_kern_pca[y == 1, 0], x_sk_kern_pca[y == 1, 1], c='blue', marker='o', alpha=0.5)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()
