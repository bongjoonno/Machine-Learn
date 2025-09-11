from model_imports import LinearRegression, train_test_split, scale_data, r_squared
from data_imports import insurance_x, insurance_y

insurance_x_train, insurance_y_train, insurance_x_test, insurance_y_test = train_test_split(insurance_x, insurance_y)

insurance_x_train, insurance_x_test = scale_data(insurance_x_train, insurance_x_test, ['age', 'bmi', 'children'])

model = LinearRegression()

model.train(insurance_x_train, insurance_y_train, 1_000, 0.05)
y_pred = model.predict(insurance_x_test)

accuracy = r_squared(y_pred, insurance_y_test)
print(accuracy)