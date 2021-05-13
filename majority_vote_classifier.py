"""
Creating a majority vote classifier class for ensemble learning
Alternative sklearn.ensemble.VotingClassifier
"""

from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.preprocessing import LabelEncoder
import six  # pip install six to solve deprecation warning
from sklearn.pipeline import _name_estimators
import numpy as np
import operator


class MajorityVoteClassifier(BaseEstimator, ClassifierMixin):
    """A majority vote ensemble classifier
    Parameters:
          classifiers : array-like, shape = [n_classifiers]. Different classifiers for the ensemble
          vote : str, {'classlabel', 'probability'}
            Default : 'classlabel'
            If 'classlabel' the prediction is based of argmax of class labels. Else if 'probability',
            the argmax of the sum of probabilities is used to predict the class label (recommended
            for calibrated classifiers).
          weights : array-like, shape = [n_classifiers]
            Optional, default: None
            If a list of int or float values provided, the classifiers are weighted by importance.
            Uses uniform weights if weights=None
    """

    def __init__(self, classifiers, vote='classlabel', weights=None):
        self.classifiers = classifiers
        self.named_classifiers = {key: value for key, value in _name_estimators(classifiers[0])}
        self.vote = vote
        self.weights = weights

    def fit(self, x, y):
        """Fit classifiers
        Parameters:
              x: {array-like, sparse matrix}, shape = [n_samples, n_features]
                Matrix of training samples.
              y: array-like, shape = [n_samples]
                Vector of target class labels.
        Returns:
              self : object
        """
        # Use LabelEncoder to ensure class labels start with 0, which is important for the
        # np.argmax call in self.predict
        self.labelenc_ = LabelEncoder()
        self.labelenc_.fit(y)
        self.classes_ = self.labelenc_.classes_
        self.classifiers_ = []

        for clf in self.classifiers[0]:  # Take first element of classifier 2d array
            fitted_classifier = clone(clf).fit(x, self.labelenc_.transform(y))
            self.classifiers_.append(fitted_classifier)

        return self

    def predict(self, x):
        """Predict class labels for x
        Parameters:
            x: {array-like, sparse matrix}, shape = [n_samples, n_features]
                Matrix of training samples.
        Returns:
              maj_vote : array-like, shape = [n_samples]
                Predicted class labels.
        """
        if self.vote == 'probability':
            maj_vote = np.argmax(self.predict_proba(x), axis=1)
        else:  # 'classlabel' vote
            # return results from classifier.predict calls
            predictions = np.asarray([clf.predict(x) for clf in self.classifiers_]).T
            maj_vote = np.apply_along_axis(lambda i: np.argmax(np.bincount(i, weights=self.weights)),
                                           axis=1, arr=predictions)

        maj_vote = self.labelenc_.inverse_transform(maj_vote)
        return maj_vote

    def predict_proba(self, x):
        """Predict class probabilities for x
        Parameters:
            x: {array-like, sparse matrix}, shape = [n_samples, n_features]
                Training vectors, where n_samples is the number of samples and n_features is
                the number of features
        Returns:
              avg_proba: array-like, shape = [n_samples, n_features]
                Weighted average probability for each class per sample.
        """
        probabilities = np.asarray([clf.predict_proba(x) for clf in self.classifiers_])
        avg_proba = np.average(probabilities, axis=0, weights=self.weights)
        return avg_proba

    def get_params(self, deep=True):
        """Get classifier parameter names for GridSearch"""
        if not deep:
            return super(MajorityVoteClassifier, self).get_params(deep=False)

        out = self.named_classifiers.copy()
        for name, step in six.iteritems(self.named_classifiers):
            for key, value in six.iteritems(step.get_params(deep=True)):
                out['{0}__{1}'.format(name, key)] = value
        return out
