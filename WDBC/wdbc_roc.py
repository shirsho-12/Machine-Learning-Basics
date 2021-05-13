"""
Receiver Operator Characteristic(ROC) graph used to select models for classification based on their performance
Area Under the Curve(AUC) characterizes the performance of the classification model
"""

from sklearn.metrics import roc_curve, auc, roc_auc_score, accuracy_score
from sklearn.model_selection import StratifiedKFold
from wdbc_common_funcs import wdbc_initializer, lr_pipeline, svc_pipeline
from scipy import interp
import numpy as np
import matplotlib.pyplot as plt

x_train, x_test, y_train, y_test = wdbc_initializer()
pipe_lr = lr_pipeline()
pipe_svc = svc_pipeline()

x_train2 = x_train[:, [4, 14]]
cv = StratifiedKFold(n_splits=3, random_state=1)
mean_tpr = 0.0                         # TPR - True Positive Rate, i.e. True Positives / All Positives(FN + TP)
mean_fpr = np.linspace(0, 1, 100)                # False Positive Rate - FP / (FP + TN)
all_tpr = []

for i, (train, test) in enumerate(cv.split(x_train, y_train)):
    probabilities = pipe_lr.fit(x_train2[train], y_train[train]).predict_proba(x_train2[test])
    fpr, tpr, thresholds = roc_curve(y_train[test], probabilities[:, 1], pos_label=1)
    mean_tpr += interp(mean_fpr, fpr, tpr)         # Total calculated from sum of all 3 k fold values
    mean_tpr[0] = 0.0                              # Start value remains 0
    roc_auc = auc(fpr, tpr)                        # Area under curve calculation
    plt.plot(fpr, tpr, lw=1, label="ROC fol {0} (area = {1:.2f}".format(i+1, roc_auc))

plt.plot([0, 1], [0, 1], linestyle='--', color=(0.6, 0.6, 0.6), label='random guessing')

mean_tpr /= cv.n_splits                             # Total TPR divided by number of folds  to get average
mean_tpr[-1] = 1.0                                  # Final value of TPR
mean_auc = auc(mean_fpr, mean_tpr)
plt.plot(mean_fpr, mean_tpr, 'k--', label='mean ROC (area = {0:.2f})'.format(mean_auc), lw=2)
plt.plot([0, 0, 1], [0, 1, 1], lw=2, linestyle=':', color='black', label='perfect performance')

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('false positive rate')
plt.ylabel('true positive rate')
plt.title('Receiver Operator Characteristic')
plt.legend(loc="lower right")
plt.show()

"""Direct output of ROC AUC score and Accuracy on a SVM pipeline using sklearn.metrics functions"""
pipe_svc = pipe_svc.fit(x_train2, y_train)
y_pred2 = pipe_svc.predict(x_test[:, [4, 14]])
print("ROC AUC: {0:.3f}".format(roc_auc_score(y_true=y_test, y_score=y_pred2)))
print("Accuracy: {0:.3f}".format(accuracy_score(y_true=y_test, y_pred=y_pred2)))
