from r_squared import r_squared
from linear_regression import linear_regression
from logistic_regression import logistic_regression
from train_test_split import train_test_split
from imports import np
from test_data import x, y
from sklearn.preprocessing import StandardScaler

x_train, y_train, x_test, y_test = train_test_split(x, y)

scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

print(x_test)