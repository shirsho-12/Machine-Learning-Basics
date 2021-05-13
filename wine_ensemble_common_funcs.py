"""
Refactoring of the initializer functions used for ensemble plots using the wine dataset
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score


def wine_init():
    df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None)
    df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash',
                       'Magnesium', 'Total phenols', 'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
                       'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline']

    df_wine = df_wine[df_wine['Class label'] != 1]   # wine classes 2 and 3 taken
    y = df_wine['Class label'].values
    x = df_wine[['Alcohol', 'Hue']].values        # wine characteristics alcohol and hue taken

    le = LabelEncoder()
    y = le.fit_transform(y)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.40, random_state=1)
    return x_train, x_test, y_train, y_test


def tree_compare_plot(classifiers, labels):
    x_train, x_test, y_train, y_test = wine_init()
    for clf, label in zip(classifiers, labels):
        clf = clf.fit(x_train, y_train)
        y_train_pred = clf.predict(x_train)
        y_test_pred = clf.predict(x_test)
        clf_train = accuracy_score(y_train, y_train_pred)
        clf_test = accuracy_score(y_test, y_test_pred)
        print("{2} train/test accuracies: {0:.3f}/{1:.3f}".format(clf_train, clf_test, label))

    x_min, x_max = x_train[:, 0].min() - 1, x_train[:, 0].max() + 1
    y_min, y_max = x_train[:, 1].min() - 1, x_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
    f, ax_arr = plt.subplots(nrows=1, ncols=2, sharex='col', sharey='row', figsize=(8, 3))

    for index, clf, title in zip([0, 1], classifiers, labels):
        clf = clf.fit(x_train, y_train)
        z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        z = z.reshape(xx.shape)

        ax_arr[index].contourf(xx, yy, z, alpha=0.3)
        ax_arr[index].scatter(x_train[y_train == 0, 0], x_train[y_train == 0, 1],
                              c='blue', marker='^', s=50)

        ax_arr[index].scatter(x_train[y_train == 1, 0], x_train[y_train == 1, 1],
                              c='red', marker='o', s=50)
        ax_arr[index].set_title(title)
        ax_arr[index].tick_params(top=True, right=True)

    ax_arr[0].set_ylabel('Alcohol', fontsize=12)
    plt.text(10.2, -1.2, s='Hue', ha='center', va='center', fontsize=12)
    plt.show()
