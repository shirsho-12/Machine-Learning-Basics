# Refactoring the iris dataset initializer and pipeline creation for ensemble learning

import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
from majority_vote_classifier import MajorityVoteClassifier


def iris_init():
    """Initializer for iris dataset ensemble learning with training and testing splits at 50% of data"""
    iris = datasets.load_iris()
    X, y = iris.data[50:, [1, 2]], iris.target[50:]
    le = LabelEncoder()
    y = le.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1)
    return X_train, X_test, y_train, y_test


def classifier_init():
    """Initializes pipelines and ensemble learning classifiers"""
    clf_1 = LogisticRegression(penalty='l2', solver='liblinear', C=0.001, random_state=0)
    clf_2 = DecisionTreeClassifier(max_depth=1, criterion='entropy', random_state=0)
    clf_3 = KNeighborsClassifier(n_neighbors=1, metric='minkowski', p=2)
    # Decision Trees are scale invariant, hence do not need standardisation

    pipe_1 = Pipeline([['sc', StandardScaler()], ['clf', clf_1]])
    pipe_3 = Pipeline([['sc', StandardScaler()], ['clf', clf_3]])
    mv_clf = MajorityVoteClassifier(classifiers=[[pipe_1, clf_2, pipe_3]])
    return pipe_1, clf_2, pipe_3, mv_clf

