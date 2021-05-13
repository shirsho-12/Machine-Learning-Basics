"""
Creating a confusion matrix to present the number of correct and misclassifications in the form of a square
matrix using sklearn.
"""

from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from wdbc_common_funcs import wdbc_initializer, svc_pipeline
import matplotlib.pyplot as plt
x_train, x_test, y_train, y_test = wdbc_initializer()
pipe_svc = svc_pipeline()

pipe_svc.fit(x_train, y_train)
y_pred = pipe_svc.predict(x_test)
conf_mat = confusion_matrix(y_true=y_test, y_pred=y_pred)
# print(conf_mat)

fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(conf_mat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(conf_mat.shape[0]):
    for j in range(conf_mat.shape[1]):
        ax.text(x=j, y=i, s=conf_mat[i, j], va='center', ha='center')
plt.title("Predicted Label")
plt.ylabel("True Label")
plt.show()

"""Precision, recall and F1 scores"""

print("Precision: {0:.3f}".format(precision_score(y_true=y_test, y_pred=y_pred)))
print("Recall: {0:.3f}".format(recall_score(y_true=y_test, y_pred=y_pred)))
print("F1: {0:.3f}".format(f1_score(y_true=y_test, y_pred=y_pred)))