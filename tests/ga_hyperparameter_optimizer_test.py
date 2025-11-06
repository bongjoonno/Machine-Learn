from src.machine_learn.data_manipulation import train_test_validate_split
from src.machine_learn.models.logistic_regression import LogisticRegression
from src.machine_learn.data_imports.breast_cancer_data import breast_cancer_x, breast_cancer_y
from src.machine_learn.data_manipulation.train_test_validate_split import train_test_validate_split
from src.machine_learn.genetic_algorithms.ga_hyperparameter_optimizer import GAHParamOptimizer
from src.machine_learn.imports import StandardScaler
from src.machine_learn.metrics.categorical_accuracy import categorical_accuracy

def ga_hyperparameter_optimizer_test():
    x_train, y_train, x_test, y_test, x_val, y_val = train_test_validate_split(breast_cancer_x, breast_cancer_y)
    scaler = StandardScaler()
    
    x_train = scaler.fit_transform(x_train)

    x_test = scaler.transform(x_test)



    logistic_regression_model = LogisticRegression()
    ga_hparameter_optimizer = GAHParamOptimizer()
    
    ga_hparameter_optimizer.optimize(logistic_regression_model, x_val, y_val)


# generate initial population of random (epoch, learning_rate) tuple pairs
# random epochs in range (1, 100_000) and learning_rate from (0.000001, 0.5)
# evaluate each by training and getting fitness function


# ughh need validation set...
# Training:   ~65%
# Validation: ~15%
# Testing:    ~20%