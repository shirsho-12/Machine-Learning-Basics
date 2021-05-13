"""
Stratified K-Fold cross-validation to split the training set into n_splits smaller sets and iterating over them to
get an improved accuracy. The .split method is used to add the data into the k_fold class
"""

from sklearn.model_selection import StratifiedKFold, cross_val_score
from wdbc_common_funcs import wdbc_initializer, lr_pipeline
import numpy as np

x_train, x_test, y_train, y_test = wdbc_initializer()
pipe_lr = lr_pipeline()

# pipe_lr.fit(x_train, y_train)
# print('Test accuracy: {0:.3f}'.format(pipe_lr.score(x_test,y_test)))

k_fold = StratifiedKFold(n_splits=10, random_state=1)

"""
# Manual score calculation

scores = []
for k, (train, test) in enumerate(k_fold.split(x_train, y_train)):
    pipe_lr.fit(x_train[train], y_train[train])
    score = pipe_lr.score(x_train[test], y_train[test])
    scores.append(score)
    print('Fold {0}, Class dist.: {1}, Acc: {2:.3f}'.format(k+1, np.bincount(y_train[train]), score))
"""
scores = cross_val_score(estimator=pipe_lr, X=x_train, y=y_train, cv=10, n_jobs=-1)
print("Cross validation accuracy scores:", scores)
print("Cross Validation accuracy: {0:.3f} +/- {1:.3f}".format(np.mean(scores), np.std(scores)))