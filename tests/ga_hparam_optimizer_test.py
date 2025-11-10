from src.machine_learn.imports import plt
from src.machine_learn.constants import LEARNING_RATE

from src.machine_learn.models import LogisticRegression
from src.machine_learn.data_prep import breast_cancer_x, breast_cancer_y, titanic_x, titanic_y

from src.machine_learn.data_manipulation import train_test_validate_split, scale_data
from src.machine_learn.metrics import categorical_accuracy

from src.machine_learn.genetic_algorithms import GAlrOptimizer

data = [(breast_cancer_x, breast_cancer_y), (titanic_x, titanic_y)]
def test_ga_hparam_optimizer() -> None:
    for x, y in data:
        x_train, y_train, x_test, y_test, x_val, y_val = train_test_validate_split(x, y)
        x_train, x_test, x_val = scale_data(x_train, x_test, x_val, columns_to_scale = x_train.columns)

        logistic_regression_model = LogisticRegression()
        ga_hparameter_optimizer = GAlrOptimizer()
        
        epochs_lst = [_ for _ in range(1, 300, 5)]
        base_acc = []
        optim_acc = []
        
        optimal_learning_rate = ga_hparameter_optimizer.optimize(logistic_regression_model, x_val, y_val)
        
        for epochs in epochs_lst:
            logistic_regression_model.train(x_train, y_train, epochs=epochs)
            y_pred = logistic_regression_model.predict(x_test)
            acc_w_default_hparams = categorical_accuracy(y_pred, y_test)

            logistic_regression_model.train(x_train, y_train, epochs=epochs, learning_rate = optimal_learning_rate)
            y_pred = logistic_regression_model.predict(x_test)
            acc_w_optim_hparams = categorical_accuracy(y_pred, y_test)

            base_acc.append(acc_w_default_hparams)
            optim_acc.append(acc_w_optim_hparams)
        
        
        plt.plot(epochs_lst, base_acc, label = f'lr = {LEARNING_RATE}')
        plt.plot(epochs_lst, optim_acc, label = 'lr = optimized')
        plt.xlabel('epochs')
        plt.ylabel('R2')
        plt.legend()
        plt.show()