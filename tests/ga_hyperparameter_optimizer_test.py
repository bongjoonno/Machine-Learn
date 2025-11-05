from src.machine_learn.models.linear_regression import LinearRegression
from src.machine_learn.data_imports.data_imports import insurance_x, insurance_y
from src.machine_learn.data_manipulation.data_manipulation_imports import train_test_split, scale_data
from src.machine_learn.metrics.metrics_imports import r_squared
from src.machine_learn.genetic_algorithms.ga_hyperparameter_optimizer import GAHParamOptimizer

def ga_hyperparameter_optimizer_test():
    insurance_x_train, insurance_y_train, insurance_x_test, insurance_y_test = train_test_split(insurance_x, insurance_y)

    insurance_x_train, insurance_x_test = scale_data(insurance_x_train, insurance_x_test, ['age', 'bmi', 'children'])

    linear_regression_model = LinearRegression()
    ga_hparameter_optimizer = GAHParamOptimizer()
    


    for _ in 
    linear_regression_model.train(insurance_x_train, insurance_y_train)
    y_pred = linear_regression_model.predict(insurance_x_test)

    linear_regression_r_squared = r_squared(y_pred, insurance_y_test)
    



# generate initial population of random (epoch, learning_rate) tuple pairs
# random epochs in range (1, 100_000) and learning_rate from (0.000001, 0.5)
# evaluate each by training and getting fitness function