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
        optimal_learning_rates = []
        
        for epochs in epochs_lst:
            logistic_regression_model.train(x_train, y_train, epochs=epochs)
            y_pred = logistic_regression_model.predict(x_test)
            acc_w_default_hparams = categorical_accuracy(y_pred, y_test)

            optimal_learning_rate = ga_hparameter_optimizer.optimize(logistic_regression_model, x_val, y_val)
            optimal_learning_rates.append(optimal_learning_rate)
            
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
        
        epochs_at_convergence_base = epochs_lst[base_acc.index(max_r2_base_acc)]
        epochs_at_convergence_optim = epochs_lst[optim_acc.index(max_r2_optim_acc)]
        
        lr_at_convergence_optim = optimal_learning_rates[optim_acc.index(max_r2_optim_acc)]
        
        print(f'{epochs_at_convergence_base=}')
        print(f'{epochs_at_convergence_optim=}')
        print(f'{lr_at_convergence_optim=}')
    
        convergence_diff = epochs_at_convergence_base - epochs_at_convergence_optim
        
        print(f'{convergence_diff=}')
        
        convergence_diff_percentage = ((epochs_at_convergence_optim - epochs_at_convergence_base) / epochs_at_convergence_base) * 100
        
        if convergence_diff_percentage < 0:
            print(f'Convergence time was reduced by {abs(convergence_diff_percentage):.2f}%')
        else:
            print(f'Convergence time was increase by {abs(convergence_diff_percentage):.2f}%, you suck.')
        
        logistic_regression_model.train(x_train, y_train, epochs = epochs_at_convergence_optim, learning_rate = lr_at_convergence_optim)
        y_pred = logistic_regression_model.predict(x_test)
        final_accuracy_w_optimized_hparams = categorical_accuracy(y_pred, y_test)
        
        print(f'{final_accuracy_w_optimized_hparams=}')
        
        plt.plot(epochs_lst, base_acc, label = f'lr = {LEARNING_RATE}')
        plt.plot(epochs_lst, optim_acc, label = 'lr = optimized')
        plt.xlabel('epochs')
        plt.ylabel('R2')
        plt.legend()
        plt.show()