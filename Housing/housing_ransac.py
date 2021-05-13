"""
Applying the RANdom SAmple Consensus(RANSAC) algorithm from the scikit-learn library
on the housing dataset to make a more robust regression model compared to linear regression.
"""

from sklearn.linear_model import RANSACRegressor, LinearRegression
import numpy as np
from housing_common_funcs import df
import matplotlib.pyplot as plt

x = df[['RM']].values
y = df[['MEDV']].values

ransac = RANSACRegressor(LinearRegression(), max_trials=100, min_samples=50,
                         loss='squared_loss', residual_threshold=5.0, random_state=0)
ransac.fit(x, y)

inlier_mask = ransac.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)
line_x = np.arange(3, 10, 1)
line_y_ransac= ransac.predict(line_x[:, np.newaxis])
plt.scatter(x[inlier_mask], y[inlier_mask], c='blue', edgecolors='black',
            marker='o', label='Inliers')
plt.scatter(x[outlier_mask], y[outlier_mask], c='lightgreen', edgecolors='black',
            marker='s', label='Outliers')
plt.plot(line_x, line_y_ransac, color='red')
plt.xlabel('Average Number of rooms [RM]')
plt.ylabel('Price in $1000\'s [MEDV]')
plt.legend(loc='upper left')
plt.show()

print("Slope: {0:.3f}".format(ransac.estimator_.coef_[0][0]))
print("Intercept: {0:.3f}".format(ransac.estimator_.intercept_[0]))
