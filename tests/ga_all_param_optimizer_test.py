from src.machine_learn.imports import np, plt, tqdm, StandardScaler
from src.machine_learn.metrics import r_squared
from src.machine_learn.data_prep import salary_x, salary_y
from src.machine_learn.data_manipulation import train_test_validate_split, scale_data
from src.machine_learn.genetic_algorithms import GAOptimizer

from src.machine_learn.data_prep import linear_regression_test_data

def test_ga_all_param_optimizer():
        scaler = StandardScaler()
        ga_optimizer = GAOptimizer()
         
        for x, y, cols_to_scale in linear_regression_test_data:
                x_train, y_train, x_val, y_val, x_test, y_test = train_test_validate_split(x, y)
                x_train, x_val, x_test = scale_data(x_train, x_val, x_test, columns_to_scale=cols_to_scale)
                
                y_train = scaler.fit_transform(y_train.to_numpy().reshape(-1, 1)).flatten()
                y_val = scaler.transform(y_val.to_numpy().reshape(-1, 1)).flatten()
                
                ga_optimizer.train(x_train, y_train, mutate=False)
        
                y_pred = ga_optimizer.predict(x_val)
                
                r2 = r_squared(y_pred, y_val)
                print(f'{r2=}')
                
                plt.plot(y_pred)
                plt.plot(y_val)
                plt.legend(['y_pred', 'y_val'])
                plt.show()