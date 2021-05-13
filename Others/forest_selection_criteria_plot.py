"""
Plot to show accuracy of different selection criteria used to split data to 2 sets for decision tree creation
Classification error is the most inaccurate
Gini Index and Entropy show similar results
"""
import matplotlib.pyplot as plt
import numpy as np


# Gini Index 
def gini(p):
    return p * (1 - p) + (1 - p) * (1 - (1 - p))                       # 2 * (p - p ** 2)


# Entropy
def entropy(p):
    return -p * np.log2(p) - (1-p) * np.log2(1-p)


# Classification Error
def error(p):
    return 1 - np.max([p, 1 - p])


x = np.arange(0.0, 1.0, 0.01)
ent = [entropy(p) if p != 0 else None for p in x]                       # List comprehensions
scaled_ent = [e * 0.5 if e else None for e in ent]
err = [error(i) for i in x]
fig = plt.figure()
ax = plt.subplot(1, 1, 1)
for i, lab, ls, c in zip([ent, scaled_ent, gini(x), err],
                         ['Entropy', 'Entropy (scaled)', 'Gini Impurity/ Index', 'Misclassification Error'],
                         ['-', '-', '--', '-.'],
                         ['black', 'lightgray', 'red', 'green', 'cyan']):
    line = ax.plot(x, i, label=lab, linestyle=ls, color=c)

ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, fancybox=True, shadow=False)
plt.axhline(y=0.5, linewidth=1, color='k', linestyle='--')
plt.axhline(y=1.0, linewidth=1, color='k', linestyle='--')
plt.tick_params(right=True)   # adds ticks to the sides
plt.ylim([0.0, 1.1])
plt.xlabel('p(i=1)')
plt.ylabel('Impurity Index')
plt.show()

