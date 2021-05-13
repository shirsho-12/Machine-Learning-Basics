"""
Selecting a suitable exploratory variable for linear regression to predict MEDV using the Seaborn library.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from housing_common_funcs import df

# Pairplot of 5 different features to visualize correlation
sns.set(style='whitegrid', context='notebook')
cols = ['LSTAT', 'INDUS', 'NOX', 'RM', 'MEDV']
sns.pairplot(df[cols], height=2.5)
plt.show()

# Heatmap of 5 features's Pearson Correlation Index to see linear correlation
cm = np.corrcoef(df[cols].values.T)
sns.set(font_scale=1.2)
sns.set_palette('pastel')
heat_map = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f',
                       annot_kws={'size': 15}, yticklabels=cols, xticklabels=cols)
plt.show()

"""
Comparision with MEDV: Median value of owner-occupied homes in $1000s
LSTAT: Percentage lower status of the population: -0.74 correlation, but pairplot shows this is not linear
RM: Average number of rooms per dwelling: +0.70 correlation, more linear
Hence RM selected as the exploratory variable
"""
