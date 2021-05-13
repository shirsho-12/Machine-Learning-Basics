"""
Supervised dimensionality reduction via scikit-learn's  implementation of
Linear Discriminant Analysis(LDA) on the wine dataset to reduce 13 dimensions of data to the top 2
and a plot of decision regions of the data using logistic regression.
Comparision of discriminants of the features of data to find that with the highest amount of class discriminatory
information.
"""

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from wine_comon_funcs import wine_initializer, plot_decision_regions
import matplotlib.pyplot as plt


x_train_std, y_train, x_test_std, y_test, _ = wine_initializer()
lda = LinearDiscriminantAnalysis(n_components=2)
x_train_lda = lda.fit_transform(x_train_std, y_train)

lr = LogisticRegression().fit(x_train_lda, y_train)
plot_decision_regions(x_train_lda, y_train, classifier=lr)
plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.legend(loc='lower left')
plt.title('Training data plot')
plt.show()

"""Repeated with test data: no misclassification error"""
x_test_lda = lda.transform(x_test_std)
plot_decision_regions(x_test_lda, y_test, classifier=lr)
plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.title("Test data plot")
plt.legend(loc='lower left')
plt.show()
