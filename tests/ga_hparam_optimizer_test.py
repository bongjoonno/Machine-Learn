from src.machine_learn.imports import plt
from src.machine_learn.constants import LEARNING_RATE

from src.machine_learn.models import LinearRegression
from src.machine_learn.data_prep import student_x, student_y

from src.machine_learn.data_manipulation import train_test_validate_split, scale_data
from src.machine_learn.metrics import r_squared

from src.machine_learn.genetic_algorithms import GAlrOptimizer

def test_ga_hparam_optimizer() -> None:
    
    linear_regression_model = LinearRegression()
    ga_hparameter_optimizer = GAlrOptimizer()
        
    x_train, y_train, x_test, y_test, x_val, y_val = train_test_validate_split(student_x, student_y)
    x_train, x_val, x_test = scale_data(x_train, x_val, x_test, columns_to_scale = x_train.columns)
        
        
    optim_epochs, optim_lr = ga_hparameter_optimizer.optimize(linear_regression_model, x_train, y_train, x_val, y_val)
    
    
    linear_regression_model.train(x_train, y_train, epochs = optim_epochs, learning_rate = optim_lr)
    y_pred = linear_regression_model.predict(x_val)
    r2 = r_squared(y_pred, y_val)
    
    plt.plot(range(len(y_pred)), sorted(y_pred), label = 'Grid-Search Epoch Optimizer')
    
    print(f'{optim_epochs=}')
    print(f'{optim_lr=}')
    print(r2)
    
    linear_regression_model.train_early_stop(x_train, y_train, x_val, y_val, learning_rate = optim_lr)
    y_pred = linear_regression_model.predict(x_val)
    r2 = r_squared(y_pred, y_val)
    
    print(f'{linear_regression_model.epochs=}')
    print(f'{optim_lr=}')
    print(r2)
    
    
    plt.plot(range(len(y_pred)), sorted(y_pred), label = 'Early-Stop Epoch Optimizer')
    plt.plot(range(len(y_val)), sorted(y_val), label = 'Validation Set')
    plt.legend()
    plt.show()
    
    
        
        