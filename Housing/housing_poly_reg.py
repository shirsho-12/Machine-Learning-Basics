"""
Polynomial regression using the LSTAT feature of the housing dataset to predict MEDV values.
Regressors of degrees 1, 2, and 3 used. Higher the degree, better the R^2 score, but higher the likelihood
of overfitting.
Taking log of the MEDV against the square root of LSTAT however, gives the best results.
"""

from housing_common_funcs import df
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
import numpy as np
import matplotlib.pyplot as plt

# initialization
x = df[['LSTAT']].values
y = df['MEDV'].values
regr = LinearRegression()

# create polynomial features
quadratic = PolynomialFeatures(degree=2)
cubic = PolynomialFeatures(degree=3)

x_fit = np.arange(x.min(), x.max(), 1)[:, np.newaxis]
x_quad = quadratic.fit_transform(x)
x_cubic = cubic.fit_transform(x)

# variable initialization
y_lin_fit, y_quad_fit, y_cubic_fit = [], [], []
linear_r2, quadratic_r2, cubic_r2 = 0, 0, 0

# Condensing fits into loop
for count, y_arr, reg, x_arr, r2s, label, c, ls in zip(range(3), [y_lin_fit, y_quad_fit, y_cubic_fit],
                                                       [regr, quadratic, cubic], [x, x_quad, x_cubic],
                                                       [linear_r2, quadratic_r2, cubic_r2],
                                                       ['Linear', 'Quadratic', 'Cubic'],
                                                       ['blue', 'red', 'green'], [':', '-', '--']):
    # Fitting the data
    regr = regr.fit(x_arr, y)
    if count == 0:
        y_arr = regr.predict(x_fit)
    else:
        y_arr = regr.predict(reg.fit_transform(x_fit))
    r2s = r2_score(y, regr.predict(x_arr))
    # Result plot
    plt.plot(x_fit, y_arr, label='{0} (d={1}), R\N{SUPERSCRIPT TWO} ={2:.2f}'.format(label, count+1, r2s),
             color=c, lw=2, linestyle=ls)
    print(r2s)

# Training data scatter plot
plt.scatter(x, y, label='training points', color='lightgray')

plt.xlabel('% lower status of population [LSTAT]')
plt.ylabel('Price in $1000\'s [MEDV]')
plt.legend(loc='upper right')
plt.show()

x_log = np.log(x)
y_sqrt = np.sqrt(y)

x_fit = np.arange(x_log.min() - 1, x_log.max() + 1, 1)[:, np.newaxis]
regr = regr.fit(x_log, y_sqrt)
y_lin_fit = regr.predict(x_fit)
linear_r2 = r2_score(y_sqrt, regr.predict(x_log))
plt.scatter(x_log, y_sqrt, label='training points', color='lightgray')
plt.plot(x_fit, y_lin_fit, label='{0} (d={1}),'
                                 'R\N{SUPERSCRIPT TWO} ={2:.2f}'.format(label, 1, linear_r2),
             color='blue', lw=2)
plt.legend(loc='lower left')
plt.xlabel('log(% lower status of population [LSTAT])')
plt.ylabel('$\sqrt{Price \; in \; \$1000\'s [MEDV]}$')
plt.show()
