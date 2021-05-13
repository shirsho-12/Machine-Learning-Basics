"""
L1 regularization to get sparse solutions to get more well-defined decision regions
"""

import matplotlib.pyplot as plt
import numpy as np
from wine_comon_funcs import wine_initializer
from sklearn.linear_model import LogisticRegression

x_train_std, y_train, x_test_std, y_test, columns = wine_initializer()

lr = LogisticRegression(penalty='l1', C=0.1, multi_class='auto', solver='liblinear')        # penalty = L 1
lr.fit(x_train_std, y_train)

"""
print('Training accuracy: ',lr.score(x_train_std, y_train))
print('Test accuracy: ',lr.score(x_test_std, y_test))
print(lr.intercept_)
print(lr.coef_)
"""

fig = plt.figure()
ax = plt.subplot(111)
colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black',
          'pink', 'lightgreen', 'lightblue', 'gray', 'indigo', 'orange']

weights, params = [], []
for c in np.arange(-4, 6):
    lr = LogisticRegression(penalty='l1', C=10.0**c, random_state=0, multi_class='auto', solver='liblinear')
    lr.fit(x_train_std, y_train)
    weights.append(lr.coef_[1])
    params.append(10.0**c)

weights = np.array(weights)
for column, color in zip(range(weights.shape[1]), colors):
    plt.plot(params, weights[:, column], label=columns[column + 1], color=color)
plt.axhline(0, color='black', linestyle='--', linewidth=3)
plt.xlim([10.0**(-5), 10**5])
plt.ylabel("Weight Coefficient")
plt.xlabel("C")
plt.xscale('log')
plt.tick_params(top=True, right=True)   # adds ticks to the sides
plt.legend(loc='lower left')
# ax.legend(loc='upper left', bbox_to_anchor=(0.88, 1.03), ncol=1, fancybox=True)
plt.show()
