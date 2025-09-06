from imports import np, pd, StandardScaler

from r_squared import r_squared
from linear_regression import linear_regression
from logistic_regression import logistic_regression
from train_test_split import train_test_split
from categorical_accuracy import categorical_acc

from categorical_test_data import x_cat, y_cat
from continuous_test_data import x_continuous, y_continuous

from scale_data import scale_data

x_train, y_train, x_test, y_test = train_test_split(x_continuous, y_continuous)

scaler = StandardScaler()

x_train, x_test = scale_data(x_train, x_test, ['age', 'bmi', 'children'])

#y_pred = linear_regression(x_train, y_train, x_test, learning_rate = 0.1)

#print(r_squared(y_pred, y_test))