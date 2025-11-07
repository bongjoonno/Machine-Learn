from src.machine_learn.data_manipulation import train_test_validate_split
from src.machine_learn.models.logistic_regression import LogisticRegression
from src.machine_learn.data_imports.breast_cancer_data import breast_cancer_x, breast_cancer_y
from src.machine_learn.data_manipulation.train_test_validate_split import train_test_validate_split
from src.machine_learn.genetic_algorithms.ga_hyperparameter_optimizer import GAHParamOptimizer
from src.machine_learn.genetic_algorithms.ga_hparam_optim_lr_only import GAHParamOptim
from src.machine_learn.imports import StandardScaler
from src.machine_learn.metrics.categorical_accuracy import categorical_accuracy
from src.machine_learn.imports import plt
from src.machine_learn.constants import LEARNING_RATE

def ga_hyperparameter_optimizer_test():
    x_train, y_train, x_test, y_test, x_val, y_val = train_test_validate_split(breast_cancer_x, breast_cancer_y)
    scaler = StandardScaler()
    
    x_train = scaler.fit_transform(x_train)

    x_test = scaler.transform(x_test)



    logistic_regression_model = LogisticRegression()
    ga_hparameter_optimizer = GAHParamOptim()
    
    epochs_lst = [_ for _ in range(1, 1_000, 20)]
    base_acc = []
    optim_acc = []
    
    for epochs in epochs_lst:
        logistic_regression_model.train(x_train, y_train, epochs=epochs)
        y_pred = logistic_regression_model.predict(x_test)
        acc_w_default_hparams = categorical_accuracy(y_pred, y_test)

        optimal_learning_rate = ga_hparameter_optimizer.optimize(logistic_regression_model, x_val, y_val)

        logistic_regression_model.train(x_train, y_train, epochs=epochs, learning_rate = optimal_learning_rate)
        y_pred = logistic_regression_model.predict(x_test)
        acc_w_optim_hparams = categorical_accuracy(y_pred, y_test)

        base_acc.append(acc_w_default_hparams)
        optim_acc.append(acc_w_optim_hparams)
    
    total_base_acc = sum(base_acc)
    total_optim_acc = sum(optim_acc)
    
    max_r2_base_acc = max(base_acc)
    max_r2_optim_acc = max(optim_acc)
    
    if max_r2_base_acc > max_r2_optim_acc:
        print('optimizer failed to converge...')
        
    print(f'{max_r2_base_acc=}')
    print(f'{max_r2_optim_acc=}')
    
    print(f'{total_base_acc=}')
    print(f'{total_optim_acc=}')
    
    epochs_at_convergence_base = base_acc[base_acc.index(max_r2_base_acc)]
    epochs_at_convergence_optim = optim_acc[optim_acc.index(max_r2_optim_acc)]
    
    print(f'{epochs_at_convergence_base=}')
    print(f'{epochs_at_convergence_optim=}')
   
    convergence_diff = epochs_at_convergence_base - epochs_at_convergence_optim
    
    print(f'{convergence_diff=}')
    
    
    plt.plot(epochs_lst, base_acc, label = 'lr = 0.1')
    plt.plot(epochs_lst, optim_acc, label = 'lr = optimized')
    plt.xlabel('epochs')
    plt.ylabel('R2')
    plt.legend()
    plt.show()
