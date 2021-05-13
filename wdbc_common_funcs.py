"""
Refactoring the initializations of the Wisconsin Breast Cancer dataset
and the logistic regression pipeline in one file
"""
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import  Pipeline
from sklearn.svm import SVC


def wdbc_initializer(arg=''):
    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data',
                     header=None)

    x = df.iloc[:, 2:].values
    y = df.iloc[:, 1].values
    le = LabelEncoder()
    y = le.fit_transform(y)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=1)
    if arg == 'cross_val':
        return x, y, x_train, x_test, y_train, y_test
    return x_train, x_test, y_train, y_test


"""
Forming a pipeline using sklearn's Pipeline class to chain the data compression, feature selection and logistic
regression steps together.
"""


def lr_pipeline():
    pipe_lr = Pipeline([('scl', StandardScaler()),
                        ('pca', PCA(n_components=2)),
                        ('lr', LogisticRegression(random_state=1, solver='liblinear'))])

    return pipe_lr


def svc_pipeline():
    pipe_svc = Pipeline([('scl', StandardScaler()),
                         ('lf', SVC(random_state=1))])
    return pipe_svc

