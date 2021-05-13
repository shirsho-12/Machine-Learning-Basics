"""
Implementation of a RBF kernel PCA for dimensionality reduction using SciPy and NumPy helper functions
"""

from scipy.spatial.distance import pdist, squareform
from scipy import exp
from scipy.linalg import eigh
import numpy as np


def rbf_kernel_pca(x, gamma, n_components):
    """
    RBF Kernel PCA implementation
    
    :param x: {NumPy Array}, shape = [n_samples, n_features]
    :param gamma: float - Tuning parameter of the RBF Kernel
    :param n_components: int - Number of principal components to return
    :return: x_pc {NumPy Array}, shape = [n_samples, k_features]. Projected Dataset
             lambdas {list} - eigenvalues.
    """

    # Calculate pairwise squared Euclidean distances in the MxN dimensional dataset.
    sq_dists = pdist(x, 'sqeuclidean')

    # Convert pairwise distances into a square matrix
    mat_sq_dists = squareform(sq_dists)

    # Compute the symmetrical kernel matrix
    k = exp(-gamma * mat_sq_dists)

    # Center the kernel matrix
    n = k.shape[0]
    one_n = np.ones((n, n)) / n
    k = k - k.dot(one_n) + one_n.dot(k).dot(one_n)

    # Obtaining eigenpairs from the centered kernel matrix
    # numpy.eigh returns them in sorted order
    eig_vals, eig_vecs = eigh(k)

    # Collect the top k eigenvectors (projected samples)
    x_pc = np.column_stack((eig_vecs[:, -i]) for i in range(1, n_components + 1))

    # Collect the corresponding eigenvalues
    lambdas = [eig_vals[-i] for i in range(1, n_components + 1)]

    return x_pc, lambdas
