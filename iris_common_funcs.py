# Refactoring the plot_decision_regions to one file
# Natural Language Processing task

from matplotlib.colors import ListedColormap
from matplotlib import pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import datasets
# from sklearn.metrics import accuracy_score


def initializer(req_data='std_dev'):
    # initializer for iris dataset variables used to train and test linear models
    iris = datasets.load_iris()
    X = iris.data[:, [2, 3]]
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    # n_iter deprecated- max_iter used instead
    # y_pred = ppn.predict(X_test_std)
    # print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
    y_combined = np.hstack((y_train, y_test))
    if req_data == 'std_dev':
        sc = StandardScaler()
        sc.fit(X_train)
        X_train_std = sc.transform(X_train)
        X_test_std = sc.transform(X_test)
        X_combined_std = np.vstack((X_train_std, X_test_std))
        return X_train_std, y_train, X_combined_std, y_combined

    if req_data == 'val':
        X_combined = np.vstack((X_train, X_test))
        return X_train, y_train, X_combined, y_combined


def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):

    # marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
    # print(xx1.shape, "\n\n", xx2.shape)
    z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    z = z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for index, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y==cl, 0], y=X[y==cl, 1], alpha=0.8,
                    c=cmap(index), marker=markers[index], label=cl, edgecolor='black')

    # Marking the test data with circles
    if test_idx:
        X_test, y_test = X[test_idx, :], y[test_idx]
        plt.scatter(X_test[:, 0], X_test[:, 1], c='', edgecolor='black', alpha=1.0,
                    linewidth=1, marker='o', s=55, label='test set')
