"""
Using the iris dataset to compare different classification methods and Majority Voting (an ensemble method)
in tandem. Majority voting makes use of the splitting of a dataset into smaller subsets and makes predictions
on the data. The result of the multiple predictions are compared and the best results are taken as a form of
plurality voting. The method is more accurate than simple pipelines when the error rate is below 0.5.
"""
from iris_ensemble_common_funcs import iris_init, classifier_init
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from itertools import product
from sklearn.model_selection import GridSearchCV

x_train, x_test, y_train, y_test = iris_init()
pipe_1, clf_2, pipe_3, mv_clf = classifier_init()
all_clf = [pipe_1, clf_2, pipe_3, mv_clf]
clf_labels = ['Logistic Regression', 'Decision Tree', 'KNN', 'Majority Voting']

sc = StandardScaler()
x_train_std = sc.fit_transform(x_train)

x_min, x_max = x_train_std[:, 0].min() - 1, x_train_std[:, 0].max() + 1
y_min, y_max = x_train_std[:, 1].min() - 1, x_train_std[:, 1].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
f, ax_arr = plt.subplots(nrows=2, ncols=2, sharex='col', sharey='row', figsize=(7, 5))

for index, clf, title in zip(product([0, 1], [0, 1]), all_clf, clf_labels):
    clf = clf.fit(x_train_std, y_train)
    z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    z = z.reshape(xx.shape)

    ax_arr[index[0], index[1]].contourf(xx, yy, z, alpha=0.3)
    ax_arr[index[0], index[1]].scatter(x_train_std[y_train == 0, 0], x_train_std[y_train == 0, 1],
                                       c='blue', marker='^', s=50)

    ax_arr[index[0], index[1]].scatter(x_train_std[y_train == 1, 0], x_train_std[y_train == 1, 1],
                                       c='red', marker='o', s=50)
    ax_arr[index[0], index[1]].set_title(title)
    ax_arr[index[0], index[1]].tick_params(top=True, right=True)

plt.text(-3.5, -4.5, s='Sepal Width(standardized)', ha='center', va='center', fontsize=12)
plt.text(-10.5, 4.5, s='Petal Width(standardized)', ha='center', va='center', fontsize=12, rotation=90)
plt.show()

"""Using GridSearchCV to tune the inverse regularisation parameter C of logistic regression"""

params = {'decisiontreeclassifier__max_depth': [1, 2], 'pipeline-1__clf__C': [0.001, 0.1, 100]}
grid = GridSearchCV(estimator=mv_clf, param_grid=params, cv=10, scoring='roc_auc')
grid.fit(x_train, y_train)
print(pd.DataFrame(grid.cv_results_))
print('Best parameters: ',grid.best_params_, '\n Accuracy: {0:.2f}'.format(grid.best_score_))
"""
Trial of outputting individual parameters used failed, however mean scores and their standard deviations
came out fine.

for param, mean_score, std_score in zip(grid.cv_results_['params'], grid.cv_results_['mean_test_score'],
                                        grid.cv_results_['std_test_score']):
    print('{0:.3f}+/-({1:.2f}) {2}'.format(mean_score, std_score, params))
"""
