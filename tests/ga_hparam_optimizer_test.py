from src.machine_learn.imports import plt
from src.machine_learn.constants import LEARNING_RATE

from src.machine_learn.models import LogisticRegression
from src.machine_learn.data_prep import student_x, student_y

from src.machine_learn.data_manipulation import train_test_validate_split, scale_data
from src.machine_learn.metrics import categorical_accuracy

from src.machine_learn.genetic_algorithms import GAlrOptimizer

data = [(student_x, student_y)]
def test_ga_hparam_optimizer() -> None:
    for x, y in data:
        x_train, y_train, x_test, y_test, x_val, y_val = train_test_validate_split(x, y)
        x_train, x_test, x_val = scale_data(x_train, x_test, x_val, columns_to_scale = x_train.columns)

        logistic_regression_model = LogisticRegression()
        ga_hparameter_optimizer = GAlrOptimizer()
        
        epochs_lst = [_ for _ in range(1, 300, 5)]
        
        optimal_learning_rate = ga_hparameter_optimizer.optimize(logistic_regression_model, x_val, y_val)
        
        accuracies = []
        
        for epochs in epochs_lst:
            logistic_regression_model.train(x_train, y_train, epochs = epochs, learning_rate = optimal_learning_rate)
            y_pred = logistic_regression_model.predict(x_val)
            accuracies.append(categorical_accuracy(y_pred, y_val))
        
        
        max_accuracy_epochs = epochs_lst[accuracies.index(max(accuracies))]
        print(max_accuracy_epochs)
        
        