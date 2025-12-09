from src.machine_learn.imports import np, plt, StandardScaler
from src.machine_learn.metrics import r_squared
from src.machine_learn.data_prep import salary_x, salary_y
from src.machine_learn.data_manipulation import train_test_validate_split, scale_data
from src.machine_learn.genetic_algorithms import ga_optimize_params

from src.machine_learn.data_prep import (salary_x, salary_y, salary_cols_to_scale,
                                         student_x, student_y, student_cols_to_scale, 
                                         car_price_x, car_price_y, car_price_cols_to_scale)

data = [(salary_x, salary_y, salary_cols_to_scale), 
        (student_x, student_y, student_cols_to_scale) ,
        (car_price_x, car_price_y, car_price_cols_to_scale)]

def test_ga_all_param_optimizer():
        scaler = StandardScaler()
        param_bounds = np.random.uniform(-2, 0, 100)
        param_bounds_dict = {}
         
        for bound in param_bounds:  
                accs = []    
                for _ in range(10):
                        for x, y, cols_to_scale in data:
                                x_train, y_train, x_val, y_val, x_test, y_test = train_test_validate_split(x, y)
                                x_train, x_val, x_test = scale_data(x_train, x_val, x_test, columns_to_scale=cols_to_scale)
                                
                                x_train = x_train.to_numpy()
                                x_val = x_val.to_numpy()
                                
                                x_train = np.column_stack((np.ones(len(x_train)), x_train))
                                x_val = np.column_stack((np.ones(len(x_val)), x_val))
                                
                                y_train = scaler.fit_transform(y_train.to_numpy().reshape(-1, 1)).flatten()
                                y_val = scaler.transform(y_val.to_numpy().reshape(-1, 1)).flatten()
                                
                                theta = ga_optimize_params(x_train, y_train, param_lower_bound = bound, mutate=False)
                        
                                
                                y_pred = x_val @ theta
                        
                                accs.append(r_squared(y_pred, y_val))

                param_bounds_dict[bound] = np.mean(accs)
                                

        print(param_bounds_dict)  