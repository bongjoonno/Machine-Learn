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
    x_train, x_test, x_val = scale_data(x_train, x_test, x_val, columns_to_scale = x_train.columns)
        
        
    optim_epochs, optim_lr = ga_hparameter_optimizer.optimize(linear_regression_model, x_train, y_train, x_val, y_val)
        
    print(optim_epochs, optim_lr)
    
    
    linear_regression_model.train(x_train, y_train, epochs = optim_epochs, learning_rate = optim_lr)
    y_pred = linear_regression_model.predict(x_val)
    accuracy = r_squared(y_pred, y_val)
    
    print(accuracy)
    
    plt.plot(range(len(y_pred)), y_pred, alpha = 0.6)
    plt.plot(range(len(y_test)), y_test, alpha = 0.6)
    plt.show()
    
    
        
        