from imports import np, pd, StandardScaler

from r_squared import r_squared
from linear_regression import linear_regression
from logistic_regression import logistic_regression
from train_test_split import train_test_split
from categorical_accuracy import categorical_acc

from categorical_test_data import x_cat, y_cat
from continuous_test_data import x_continuous, y_continuous

x_train, y_train, x_test, y_test = train_test_split(x_continuous, y_continuous)

scaler = StandardScaler()

x_train[['age', 'bmi', 'children']] = scaler.fit_transform(x_train.loc[:, ['age', 'bmi', 'children']])
print(x_train)