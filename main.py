from r_squared import r_squared
from linear_regression import linear_regression
from train_test_split import train_test_split

x = [1,2,3,4,5,6,7,8,9,10]
y = [1,2,3,4,5,6,7,8,9,10]

x_train, y_train, x_test, y_test = train_test_split(x, y)

y_pred = linear_regression(x_train, y_train, x_test, y_test)

r_2 = r_squared(y_pred, y_test)

print(r_2)


