"""
Using decision trees to predict MEDV values of houses using the LSTAT feature of the housing dataset. Better
results than polynomial regression obtained.
Using random forests, all the features of the housing dataset are used to predict MEDV values,
results are much more accurate than that obtained by linear regression.
"""
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from housing_common_funcs import df, lin_reg_plot, housing_initializer
import matplotlib.pyplot as plt

x = df[['LSTAT']].values
y = df['MEDV'].values
tree = DecisionTreeRegressor(max_depth=3).fit(x, y)
sort_index = x.flatten().argsort()
lin_reg_plot(x[sort_index], y[sort_index], tree)

plt.xlabel('% lower status of population [LSTAT]')
plt.ylabel('Price in $1000\'s [MEDV]')
plt.show()

x_train, x_test, y_train, y_test = housing_initializer(test=0.4, r_state=1)
forest = RandomForestRegressor(n_estimators=1000, criterion='mse', random_state=1, n_jobs=-1)
forest.fit(x_train, y_train)
y_train_pred, y_test_pred = forest.predict(x_train), forest.predict(x_test)
print('MSE train: {0:.3f} \ttest: {1:.3f}'.format(mean_squared_error(y_train, y_train_pred),
                                                  mean_squared_error(y_test, y_test_pred)))
print('R\N{SUPERSCRIPT TWO} train: {0:.3f} \ttest: {1:.3f}'.format(r2_score(y_train, y_train_pred),
                                                                   r2_score(y_test, y_test_pred)))

plt.scatter(y_train_pred, y_train_pred - y_train, c='black', s=35, alpha=0.5,
            edgecolors='black', marker='o', label='Training Data')
plt.scatter(y_test_pred, y_test_pred - y_test, c='lightgreen', s=35, alpha=0.5,
            edgecolors='black', marker='s', label='Test Data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=50, lw=2, color='red')
plt.xlim([-10, 50])
plt.show()
