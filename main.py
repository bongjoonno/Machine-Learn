from r_squared import r_squared
from linear_regression import linear_regression
from logistic_regression import logistic_regression
from train_test_split import train_test_split

x = [x for x in range(1000)]

start = 0
y = []

for _ in range(1000):
    y.append(start)
    start += 0.001

y = [1 if num >= 0.5 else 0 for num in y]

x_train, y_train, x_test, y_test = train_test_split(x, y)

y_pred = logistic_regression(x_train, y_train, x_test)

