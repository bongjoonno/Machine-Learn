from src.machine_learn.data_manipulation import train_test_validate_split
from src.machine_learn.models.logistic_regression import LogisticRegression
from src.machine_learn.data_imports.breast_cancer_data import breast_cancer_x, breast_cancer_y
from src.machine_learn.data_manipulation.train_test_validate_split import train_test_validate_split
from src.machine_learn.genetic_algorithms.ga_hyperparameter_optimizer import GAHParamOptimizer
from src.machine_learn.genetic_algorithms.ga_hparam_optim_lr_only import GAHParamOptim
from src.machine_learn.imports import StandardScaler
from src.machine_learn.metrics.categorical_accuracy import categorical_accuracy
from src.machine_learn.imports import plt
from src.machine_learn.constants import EPOCHS, LEARNING_RATE

def ga_hyperparameter_optimizer_test():
    x_train, y_train, x_test, y_test, x_val, y_val = train_test_validate_split(breast_cancer_x, breast_cancer_y)
    scaler = StandardScaler()
    
    x_train = scaler.fit_transform(x_train)

    x_test = scaler.transform(x_test)



    logistic_regression_model = LogisticRegression()
    ga_hparameter_optimizer = GAHParamOptim()
    
    
    logistic_regression_model.train(x_train, y_train)
    y_pred = logistic_regression_model.predict(x_test)
    acc_w_default_hparams = categorical_accuracy(y_pred, y_test)

    optimal_learning_rate = ga_hparameter_optimizer.optimize(logistic_regression_model, x_val, y_val)
    #0.012
    logistic_regression_model.train(x_train, y_train, learning_rate = optimal_learning_rate)
    y_pred = logistic_regression_model.predict(x_test)
    acc_w_optim_hparams = categorical_accuracy(y_pred, y_test)

    print(f'{EPOCHS=} {LEARNING_RATE=} {acc_w_default_hparams=}')
    print(f'{EPOCHS=} {optimal_learning_rate=} {acc_w_optim_hparams=}')
    

    plt.plot(ga_hparameter_optimizer.avg_fitness_scores_per_generation)
    plt.show()