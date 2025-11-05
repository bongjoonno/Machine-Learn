from machine_learn.data_manipulation import train_test_validate_split
from src.machine_learn.models.linear_regression import LinearRegression
from src.machine_learn.data_imports.data_imports import insurance_x, insurance_y
from src.machine_learn.data_manipulation.train_test_validate_split import train_test_validate_split
from src.machine_learn.metrics.metrics_imports import r_squared
from src.machine_learn.genetic_algorithms.ga_hyperparameter_optimizer import GAHParamOptimizer
from src.machine_learn.imports import StandardScaler

def ga_hyperparameter_optimizer_test():
    x_train, y_train, x_test, y_test, x_val, y_val = train_test_validate_split(insurance_x, insurance_y)
    scaler = StandardScaler()
    scaler.fit(x_train)

    x_train.loc[:, ['age', 'bmi', 'children']] = scaler.transform(x_train.loc[:, ['age', 'bmi', 'children']])
    x_test.loc[:, ['age', 'bmi', 'children']] = scaler.transform(x_test.loc[:, ['age', 'bmi', 'children']])
    x_val.loc[:, ['age', 'bmi', 'children']] = scaler.transform(x_val.loc[:, ['age', 'bmi', 'children']])

    linear_regression_model = LinearRegression()
    ga_hparameter_optimizer = GAHParamOptimizer()
    
    ga_hparameter_optimizer.optimize(linear_regression_model, x_val, y_val)
    



# generate initial population of random (epoch, learning_rate) tuple pairs
# random epochs in range (1, 100_000) and learning_rate from (0.000001, 0.5)
# evaluate each by training and getting fitness function


# ughh need validation set...
# Training:   ~65%
# Validation: ~15%
# Testing:    ~20%