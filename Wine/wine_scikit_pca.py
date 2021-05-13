"""
Unsupervised dimensionality reduction via scikit-learn's  implementation of
Principal Component Analysis(PCA) on the wine dataset to reduce 13 dimensions of data to the top 2
and a plot of decision regions of the data
"""

from wine_comon_funcs import wine_initializer, plot_decision_regions
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

x_train_std, y_train, x_test_std, y_test, _ = wine_initializer()
pca = PCA(n_components=2)
# n_components=None keeps all principal components, 
# which can be accessed via the explained_variance_ratio_ attribute. No dimensionality reduction is done.

x_train_pca, x_test_pca = pca.fit_transform(x_train_std), pca.transform(x_test_std)
# values in scikit-learn's PCA are flipped for PC2, plotted data has a 180 degree rotation/flip
# Does not affect decision region prediction capability

lr = LogisticRegression(multi_class='auto', solver='liblinear').fit(x_train_pca, y_train)
plot_decision_regions(x_train_pca, y_train, classifier=lr)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend(loc='lower left')
plt.show()

"""Repeated with test data: very small misclassification error"""
plot_decision_regions(x_test_pca, y_test, classifier=lr)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend(loc='lower left')
plt.show()