from model_imports import LinearRegression
from data_imports.data_imports import insurance_x, insurance_y
from data_manipulation_imports import train_test_split, scale_data
from metrics.metrics_imports import r_squared

insurance_x_train, insurance_y_train, insurance_x_test, insurance_y_test = train_test_split(insurance_x, insurance_y)

insurance_x_train, insurance_x_test = scale_data(insurance_x_train, insurance_x_test, ['age', 'bmi', 'children'])

linear_regression_model = LinearRegression()

linear_regression_model.train(insurance_x_train, insurance_y_train)
y_pred = linear_regression_model.predict(insurance_x_test)

linear_regression_r_squared = r_squared(y_pred, insurance_y_test)
print(linear_regression_r_squared)