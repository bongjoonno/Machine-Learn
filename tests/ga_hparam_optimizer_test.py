from src.machine_learn.imports import plt
from src.machine_learn.constants import LEARNING_RATE

from src.machine_learn.models import LinearRegression
from src.machine_learn.data_prep import (salary_x, salary_y, salary_cols_to_scale,
                                         student_x, student_y, student_cols_to_scale, 
                                         car_price_x, car_price_y, car_price_cols_to_scale, 
                                         insurance_x, insurance_y, insurance_cols_to_scale)

from src.machine_learn.data_manipulation import train_test_validate_split, scale_data
from src.machine_learn.metrics import r_squared

from src.machine_learn.genetic_algorithms import GAHParamOptimizer

data = [(salary_x, salary_y, salary_cols_to_scale), (student_x, student_y, student_cols_to_scale), (car_price_x, car_price_y, car_price_cols_to_scale), (insurance_x, insurance_y, insurance_cols_to_scale)]

def test_ga_hparam_optimizer() -> None:
    
    linear_regression_model = LinearRegression()
    
    for x, y, cols_to_scale in data:
        x_train, y_train, x_val, y_val, x_test, y_test = train_test_validate_split(x, y)
        x_train, x_val, x_test = scale_data(x_train, x_val, x_test, columns_to_scale = cols_to_scale)
        
        ga_hparameter_optimizer = GAHParamOptimizer(linear_regression_model, x_train, y_train, x_val, y_val)
            
        optim_lr = ga_hparameter_optimizer.optimize_lr()
        optim_epochs = ga_hparameter_optimizer.epochs_grid_search(optimal_lr = optim_lr)
        
        linear_regression_model.train(x_train, y_train, epochs = optim_epochs, learning_rate = optim_lr)
        grid_search_y_pred = linear_regression_model.predict(x_val)
        grid_epochs_r2 = r_squared(grid_search_y_pred, y_val)
        
        print(f'grid-search {optim_epochs=}')
        print(grid_epochs_r2)
        print('\n')
        
        linear_regression_model.train_early_stop(x_train, y_train, x_val, y_val, learning_rate = optim_lr)
        early_stop_y_pred = linear_regression_model.predict(x_val)
        early_stop_r2 = r_squared(early_stop_y_pred, y_val)
        
        print(f'early-stop-epochs = {linear_regression_model.epochs}')
        print(f'{early_stop_r2=}')
        
        
        plt.plot(grid_search_y_pred)
        plt.plot(early_stop_y_pred)
        plt.plot(y_val)
        
        plt.legend(['grid-search', 'early-stop', 'validation-set'])
        plt.show()