"""
Refactoring the wine x and y value variables initializations in one file
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from iris_common_funcs import plot_decision_regions


def wine_initializer(arg = 'std'):
    df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None)
    df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash',
                       'Magnesium', 'Total phenols', 'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
                       'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline']

    # print('Class labels', np.unique(df_wine['Class label']))
    # print(df_wine)
    x, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

    # Normalised data: useful for data to be in a bound interval. More sensitive to outliers.
    mms = MinMaxScaler()
    x_train_norm = mms.fit_transform(x_train)
    x_test_norm = mms.transform(x_test)

    # Standardization: maintains useful information about outliers, uses standard deviation. Preferred.
    sc = StandardScaler()
    if arg == 'sc_mat':
        x_train_std = sc.fit_transform(x_train)
        x_test_std = sc.fit_transform(x_test)
        return x_train_std, y_train, x_test_std, y_test, x, y
    if arg == 'std':
        x_train_std = sc.fit_transform(x_train)
        x_test_std = sc.fit_transform(x_test)
        return x_train_std, y_train, x_test_std, y_test, df_wine.columns
    if arg == 'val':
        return x_train, y_train, x_test, y_test, df_wine.columns
    else:
        print("Invalid argument")
        return 0


"""
Creating mean-vector matrices, a within class and a between class scatter matrix for LDA -
Linear Discriminant Analysis
The functions initialize the matrices
"""


def mean_vecs(x_train_std, y_train):
    """Calculating the mean vectors of the x_strain_std array"""
    mean_vectors =[]
    for label in range(1, 4):
        mean_vectors.append(np.mean(x_train_std[y_train == label], axis=0))
        # print('MV {0}:\n {1}'.format(label, mean_vectors[label - 1]))
    return mean_vectors


def within_class_sc_mat(mean_vectors, x_train_std, y_train,x,y):
    """Calculating the within-class scatter matrix from mean vectors"""
    dim = 13          # number of features
    sc_w = np.zeros((dim, dim))           # sc_w - within-class scatter matrix
    for label, m_vec in zip(range(1, 4), mean_vectors):
        """
        # Not scaled
        class_scatter = np.zeros((dim, dim))
        for row in x[y == label]:
            row, m_vec = row.reshape(dim, 1), m_vec.reshape(dim, 1)
            class_scatter += (row-m_vec).dot((row-m_vec).T)
        sc_w += class_scatter
        # print("Within-class scatter matrix: {0}x{0}".format(sc_w.shape[1]))
        """
        """Scaled"""
        class_scatter = np.cov(x_train_std[y_train == label].T)
        sc_w += class_scatter
    # print("Scaled within-class scatter matrix: {0}x{1}".format(sc_w.shape[0], sc_w.shape[1]))
    return sc_w


def between_class_sc_mat(mean_vectors, x, y, x_train_std):
    """Calculating the between-class scatter matrix from the scaled within-class scatter matrix"""
    dim = 13
    mean_overall = np.mean(x_train_std, axis=0)
    sc_b = np.zeros((dim, dim))           # sc_b - betwee-class scatter matrix
    for label, mean_vec in enumerate(mean_vectors):
        n = x[y == label + 1, :].shape[0]
        mean_vec = mean_vec.reshape(dim, 1)
        mean_overall = mean_overall.reshape(dim, 1)
        sc_b += n * (mean_vec - mean_overall).dot((mean_vec - mean_overall).T)
    # print("Between-class scatter matrix: {0}x{1}".format(sc_b.shape[0], sc_b.shape[1]))
    return sc_b


def wine_matrix_init():
    x_train_std, y_train, x_test_std, y_test, x, y = wine_initializer('sc_mat')
    np.set_printoptions(precision=4)
    mean_vectors =  mean_vecs(x_train_std, y_train)
    # unique, counts = np.unique(y_train, return_counts=True)
    # print('Class label distribution: ', dict(zip(unique, counts)))
    sc_w = within_class_sc_mat(mean_vectors, x_train_std, y_train, x,y)
    sc_b = between_class_sc_mat(mean_vectors, x, y, x_train_std)
    return mean_vectors, sc_w, sc_b


# wine_matrix_init()
