"""
Unsupervised dimensionality reduction via manual implementation of
Principal Component Analysis(PCA) on the wine dataset to reduce 13 dimensions of data to the top 2
"""
from wine_comon_funcs import wine_initializer
import numpy as np
import matplotlib.pyplot as plt


def var_plot():
    """
    Individual and cumulative explained variance plot using the wine dataset
    """
    plt.bar(range(1, 14), var_explained, alpha=0.5, align='center', label='Individual explained variance')
    plt.step(range(1, 14), cum_var_explained, where='mid', label='Cumulative explained variance')
    plt.xlabel('Principal components')
    plt.ylabel('Explained variance ratio')
    plt.legend(loc='best')
    plt.show()


def pca_plot():
    """
    Plot of 3 groups of data of reduced dimensionality and unique y_train values
    """
    colors = ['r', 'b', 'g']
    markers = ['s', 'x', 'o']
    # print(np.unique(y_train))
    for l, c, m, count in zip(np.unique(y_train), colors, markers, np.arange(1,4)):
        plt.scatter(x_train_pca[y_train == count, 0],
                    x_train_pca[y_train == count, 1],
                    c=c, label=l, marker=m)

    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.legend(loc='lower left')
    plt.show()


x_train_std, y_train, x_test_std, y_test, _ = wine_initializer()

cov_mat = np.cov(x_train_std.T)                # covariance matrix
# print(cov_mat)
eigen_vals, eigen_vecs= np.linalg.eig(cov_mat)
# print("Eigenvalues \n", eigen_vals)

total = sum(eigen_vals)
var_explained = [(i/total) for i in sorted(eigen_vals, reverse=True)]      # variance explained ratio
cum_var_explained = np.cumsum(var_explained)                               # cumulative sum of explained variances
# var_plot()

eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i]) for i in range(len(eigen_vals))]
eigen_pairs.sort(reverse=True)

w = np.hstack((eigen_pairs[0][1][:, np.newaxis], eigen_pairs[1][1][:, np.newaxis]))
# print("Matrix W: \n",w)
x_train_pca = x_train_std.dot(w)     # pca - principal component analysis
# print(x_train_pca)
pca_plot()
