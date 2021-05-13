"""
Using the handmade Linear Regression Gradient Descent on the RM (Rooms per Dwelling) feature predict MEDV
(Median Dwelling Value).
"""

from sklearn.preprocessing import StandardScaler
from housing_common_funcs import df, lin_reg_plot, housing_initializer
from linear_regression_gd import LinearRegressionGD                                # buggy
from sklearn.linear_model import LinearRegression                                  # preferred
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np

x = df[['RM']].values
y = df[['MEDV']].values


def sse_plot(model):
    plt.plot(range(1, model.n_iter + 1), model.cost_)
    plt.ylabel('SSE')
    plt.xlabel('Epoch')
    plt.show()


sc_x, sc_y = StandardScaler(), StandardScaler()
x_std, y_std = sc_x.fit_transform(x), sc_y.fit_transform(y)
lr = LinearRegressionGD()
lr.fit(x_std, y_std)


def lr_plot():
    sse_plot(lr)
    lin_reg_plot(x_std, y_std, lr)
    plt.xlabel('Average Number of rooms [RM] (standardized)')
    plt.ylabel('Price in $1000\'s [MEDV] (standardized)')
    plt.show()


# lr_plot()
# Getting the mean price
num_rooms_std = sc_x.transform(np.array([5.0]).reshape(1, -1))
price_std = lr.predict(num_rooms_std)
print("Price in $1000\'s: {0:.3f}".format(sc_y.inverse_transform(price_std)[0]))

print("Slope: {0:.3f}".format(lr.w_[1]))            
print("Intercept: {0:.3f}".format(lr.w_[0]))        # Y-intercept is 0 since the data is standardized


"""Sci-kit learn's linear regression classifier"""
s_lr = LinearRegression()


def sk_lin_reg_plot():
    s_lr.fit(x, y)
    print("Slope(sk-learn LR): {0:.3f}".format(s_lr.coef_[0][0]))
    print("Intercept(sk-learn LR): {0:.3f}".format(s_lr.intercept_[0]))
    lin_reg_plot(x, y, s_lr, colors=['yellow', 'green'])
    plt.xlabel('Average Number of rooms [RM] (standardized)')
    plt.ylabel('Price in $1000\'s [MEDV] (standardized)')
    plt.show()


# sk_lin_reg_plot()
x_train, x_test, y_train, y_test = housing_initializer()
s_lr.fit(x_train, y_train)
y_train_pred = s_lr.predict(x_train)
y_test_pred = s_lr.predict(x_test)

plt.scatter(y_train_pred, y_train_pred - y_train, c='blue',
            edgecolors='black', marker='o', label='Training Data')
plt.scatter(y_test_pred, y_test_pred - y_test, c='red',
            edgecolors='black', marker='s', label='Test Data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=50, lw=2, color='red')
plt.xlim([-10, 50])
plt.show()
"""Mean squared error(MSE) is average value of the SSE (Sum squared error) 
cost to compare linear regression models"""
print("MSE train: {0:.3f} \ttest: {1:.3f}\n".format(mean_squared_error(y_train, y_train_pred),
                                                  mean_squared_error(y_test, y_test_pred)))

"""
R^2 score is the coefficient of detemination. That is, it is standardized MSE
R^2 = 1 - SSE/SST (SST- Total Sum of Squares) 
Higher the R-squared value, better the fitting of the data (R^2 is in the range of 0 to 1 for training data,
and -1 to 1 for test data).
"""
print("R^2 train: {0:.3f} \ttest:{1:.3f}".format(r2_score(y_train, y_train_pred),
                                                  r2_score(y_test, y_test_pred)))
