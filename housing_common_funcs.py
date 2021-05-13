"""
Housing Dataset: Information about houses in the suburbs of Boston in 1978, contains 506 samples and 14 features
MEDV: Median value of owner-occupied homes in $1000s
"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data',
                 header=None, sep='\s+')
df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE',
              'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
# print(df.head())


def lin_reg_plot(x, y, model, colors=['blue', 'red']):
    plt.scatter(x, y, c=colors[0], alpha=0.5, edgecolors='black')
    plt.plot(x, model.predict(x), color=colors[1])


def housing_initializer(test=0.3, r_state=0):
    x = df.iloc[:, :-1].values
    y = df['MEDV'].values
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test, random_state=r_state)
    return x_train, x_test, y_train, y_test
