"""
SBS - Sequential Backward Selection
Greedy algorithm that removes features(characteristics or variables) from a dataset to reduce dimensionality of data,
while ensuring a minimum decay of performance.
"""

from sklearn.base import clone          # creates deepcopy of model with specified estimators(rules or functions)
from itertools import combinations
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class SBS:
    def __init__(self, estimator, k_features, scoring=accuracy_score, test_size=0.25, random_state=1):
        self.scoring = scoring
        self.estimator = clone(estimator)
        self.k_features = k_features
        self.test_size = test_size
        self.random_state = random_state

    def fit(self, x, y):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=self.test_size,
                                                            random_state=self.random_state)

        dimensions = x_train.shape[1]
        self.indices_ = tuple(range(dimensions))
        self.subsets_ = [self.indices_]
        score = self.calc_score(x_train, y_train, x_test, y_test, self.indices_)

        self.scores_ = [score]
        while dimensions > self.k_features:
            scores, subsets = [], []

            for p in combinations(self.indices_, r=dimensions-1):
                score = self.calc_score(x_train, y_train, x_test, y_test, p)
                # print(score)
                scores.append(score)
                subsets.append(p)

            best = np.argmax(scores)
            # print(best)
            self.indices_ = subsets[best]
            self.scores_.append(scores[best])
            self.subsets_.append(self.indices_)
            dimensions -= 1

        self.k_score_ = self.scores_[-1]
        return self

    def transform(self, x):
        return x[:, self.indices_]

    def calc_score(self, x_train, y_train, x_test, y_test, indices):
        self.estimator.fit(x_train[:, indices], y_train)
        y_pred = self.estimator.predict(x_test[:, indices])
        score = self.scoring(y_test, y_pred)
        return score

