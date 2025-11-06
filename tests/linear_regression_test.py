from src.machine_learn.models.linear_regression import LinearRegression
from src.machine_learn.data_imports.data_imports import insurance_x, insurance_y
from src.machine_learn.data_manipulation.train_test_split import train_test_split, scale_data
from src.machine_learn.metrics.metrics_imports import r_squared

def linear_regression_test():
    insurance_x_train, insurance_y_train, insurance_x_test, insurance_y_test = train_test_split(insurance_x, insurance_y)

    insurance_x_train, insurance_x_test = scale_data(insurance_x_train, insurance_x_test, ['age', 'bmi', 'children'])

    linear_regression_model = LinearRegression()

    linear_regression_model.train(insurance_x_train, insurance_y_train)
    y_pred = linear_regression_model.predict(insurance_x_test)

    linear_regression_r_squared = r_squared(y_pred, insurance_y_test)
    return linear_regression_r_squared