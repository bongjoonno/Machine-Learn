from src.machine_learn.imports import plt
from src.machine_learn.metrics import r_squared
from src.machine_learn.data_prep import salary_x, salary_y
from src.machine_learn.data_manipulation import train_test_validate_split, scale_data
from src.machine_learn.genetic_algorithms import optimize_weight, optimize_bias

from src.machine_learn.data_prep import (salary_x, salary_y, salary_cols_to_scale,
                                         student_x, student_y, student_cols_to_scale, 
                                         car_price_x, car_price_y, car_price_cols_to_scale, 
                                         insurance_x, insurance_y, insurance_cols_to_scale)

data = [(salary_x, salary_y, salary_cols_to_scale), (student_x, student_y, student_cols_to_scale), (car_price_x, car_price_y, car_price_cols_to_scale), (insurance_x, insurance_y, insurance_cols_to_scale)]

def test_ga_param_optimizer():
    for x, y, cols_to_scale in data:
        number_of_features = x.shape[1]
        x_train, y_train, x_val, y_val, x_test, y_test = train_test_validate_split(x, y)
        x_train, x_val, x_test = scale_data(x_train, x_val, x_test, columns_to_scale=cols_to_scale)
        
        print(x_train)
        x_train = x_train.to_numpy()
        y_train = y_train.to_numpy()
        
        theta = []
        
        for i in range(number_of_features):
            x_feature = x_train.T[i]
            print(x_feature)
            print(x_feature.shape)
            weight = optimize_weight(x_feature, y_train)
            theta.append(weight)
        
        best_preds = x_train @ theta
        
        bias = optimize_bias(best_preds, y_train)
        
        
        y_pred = (x_train @ theta) + bias
        
        print(r_squared(y_pred, y_train))
        
    
        plt.plot(y_train)
        plt.plot(y_pred)
        plt.legend(['y_true', 'y_pred'])
        plt.show()

        
# scaler should be used on y for GA optimizer
