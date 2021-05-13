"""
Hyperparameter tuning of logistic regression parameters using scikit learn's grid search clas
"""

from wdbc_common_funcs import wdbc_initializer, svc_pipeline
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.metrics import make_scorer, precision_score
# import pandas as pd

x, y, x_train, x_test, y_train, y_test = wdbc_initializer('cross_val')
pipe_svc = svc_pipeline()

param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
param_grid = [{'lf__C' : param_range,                 # Parameter C only varied for linear SVM
               'lf__kernel': ['linear']},
              {'lf__C': param_range,                  # Both parameters C amnd gamma varied for RBF Kernel SVM
               'lf__gamma': param_range,
               'lf__kernel': ['rbf']}]
"""
make_scorer function can be used to change the positive class label from 1 to any other number and scoring type
can also be changed from accuracy to any other metric such as F1 or precision
"""
pre_scorer = make_scorer(score_func=precision_score, pos_label=1, greater_is_better=True, average='micro')
gs = GridSearchCV(estimator=pipe_svc, param_grid=param_grid, iid=True, scoring=pre_scorer, cv=10, n_jobs=-1)
gs.fit(x_train, y_train)

# print("Best Score:", gs.best_score_)
# print("Best parameters", gs.best_params_)

clf = gs.best_estimator_
clf.fit(x_train, y_train)
# print("Test accuracy: {0:.3f}".format(clf.score(x_test, y_test)))
# print(pd.DataFrame.from_dict(gs.cv_results_))       # dataframe consisting of all testing combinations and data

"""
Cross-validation scores of linear SVM and Decision Trees compared to find the more suitable machine-learning 
algorithm on the dataset.
"""

scores = cross_val_score(gs, x, y, scoring='accuracy', cv=5)
print("Cross validation accuracy(SVM): {0:.3f} +/- {1:.3f}".format(np.mean(scores), np.std(scores)))

gs= GridSearchCV(estimator=DecisionTreeClassifier(random_state=0),
                 param_grid={'max_depth': [1, 2, 3, 4, 5, 6, 7, None]},
                 scoring='accuracy', cv=5)

scores = cross_val_score(gs, x, y, scoring='accuracy', cv=5)
print("Cross validation accuracy(Decision Tree): {0:.3f} +/- {1:.3f}".format(np.mean(scores), np.std(scores)))
