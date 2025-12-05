from src.machine_learn.imports import plt
from src.machine_learn.metrics import r_squared
from src.machine_learn.data_prep import salary_x, salary_y
from src.machine_learn.data_manipulation import train_test_validate_split

from src.machine_learn.genetic_algorithms import optimize_weight_and_bias_seperately

def test_ga_param_optimizer():
    x_train, y_train, x_test, y_test, x_val, y_val = train_test_validate_split(salary_x, salary_y)
    weight, bias = optimize_weight_and_bias_seperately(x_train, y_train)
    
    y_pred = (x_val * weight) + bias
    
    plt.plot(y_pred)
    plt.plot(y_val)
    plt.show()
    
    return r_squared(y_pred, y_val)
    
