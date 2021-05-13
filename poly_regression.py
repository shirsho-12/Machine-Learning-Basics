"""
Polynomial regression is adding a polynomial variable(s) to a linear regression formula to increase it's order.
It is considered a multiple linear regression model because of the linear regression coefficients w in the formula.
"""

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

x = np.array([258.0, 270.0, 294.0, 320.0, 342.0, 368.0,
              396.0, 446.0, 480.0, 586.0])[:, np.newaxis]
y = np.array([236.4, 234.4, 252.8, 298.6, 314.2, 342.2,
              360.8, 368.0, 391.2, 390.8])

lr = LinearRegression()                    # linear regression variable
pr = LinearRegression()                    # polynomial regression variable

quadratic = PolynomialFeatures(degree=2)
x_quad = quadratic.fit_transform(x)

lr.fit(x, y)
pr.fit(x_quad, y)
x_fit = np.arange(250,600, 10)[:, np.newaxis]
y_lin_fit = lr.predict(x_fit)
y_quad_fit = pr.predict(quadratic.fit_transform(x_fit))
plt.scatter(x,y, label='Training Points')
plt.plot(x_fit, y_lin_fit, label='Linear Fit', linestyle='--')
plt.plot(x_fit, y_quad_fit, label='Quadratic Fit')
plt.legend(loc="upper left")
plt.show()

y_lin_pred = lr.predict(x)
y_quad_pred = pr.predict(x_quad)
print('Training MSE linear: {0:.3f} \tquadratic: {1:.3f}'.format(mean_squared_error(y, y_lin_pred),
                                                                 mean_squared_error(y, y_quad_pred)))
print('Training R^2 score linear: {0:.3f} \tquadratic: {1:.3f}'.format(r2_score(y, y_lin_pred),
                                                                       r2_score(y, y_quad_pred)))
