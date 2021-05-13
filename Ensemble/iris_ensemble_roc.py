"""
Comparision of individual classifiers and ensemble learning using ROC AUC scores
"""

from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve, auc
from iris_ensemble_common_funcs import iris_init, classifier_init
import matplotlib.pyplot as plt

x_train, x_test, y_train, y_test = iris_init()
pipe_1, clf_2, pipe_3, mv_clf = classifier_init()
clf_labels = ['Logistic Regression', 'Decision Tree', 'KNN', 'Majority Voting']
all_clf = [pipe_1, clf_2, pipe_3, mv_clf]
colors = ['black', 'orange', 'blue', 'green']
linestyles = [':', '--', '-.', '-']
for clf, label, color, ls in zip(all_clf, clf_labels, colors, linestyles):
    y_pred = clf.fit(x_train, y_train).predict_proba(x_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_true=y_test, y_score=y_pred)
    roc_auc = auc(x=fpr, y=tpr)
    plt.plot(fpr, tpr, color=color, linestyle=ls, label='{0} (AUC = {1:.2f})'.format(label, roc_auc))

plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=2)
plt.xlim([-0.1, 1.1])
plt.ylim([-0.1, 1.1])
plt.grid()
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

print('10-fold cross validation:\n')
for clf, label in zip(all_clf, clf_labels):
    scores = cross_val_score(estimator=clf, X=x_train, y=y_train, cv=10, scoring='roc_auc')
    print("ROC AUC: {0:.2f} (+/-{1:.2f}) [{2}]".format(scores.mean(), scores.std(), label))

print(mv_clf.get_params())
