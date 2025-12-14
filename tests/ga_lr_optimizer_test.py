from src.machine_learn.imports import plt

from src.machine_learn.models import LinearRegression
from src.machine_learn.data_prep import linear_regression_test_data

from src.machine_learn.data_manipulation import train_test_validate_split, scale_data
from src.machine_learn.metrics import r_squared

from src.machine_learn.genetic_algorithms import GAlrOptimizer

def test_ga_lr_optimizer() -> None:
    linear_regression_model = LinearRegression()
    
    for x, y, cols_to_scale in linear_regression_test_data:
        x_train, y_train, x_val, y_val, x_test, y_test = train_test_validate_split(x, y)
        x_train, x_val, x_test = scale_data(x_train, x_val, x_test, columns_to_scale = cols_to_scale)
        
        ga_hparameter_optimizer = GAlrOptimizer(linear_regression_model, x_train, y_train, x_val, y_val)
            
        optim_lr = ga_hparameter_optimizer.optimize_lr()
        
        linear_regression_model.train(x_train, y_train, epochs = 1_000, learning_rate = optim_lr)
        
        y_pred = linear_regression_model.predict(x_val)
        r2 = r_squared(y_pred, y_val)
        
        print(f'{r2=}')
        
        plt.plot(range(len(y_val)), y_pred, alpha=0.3)
        plt.plot(range(len(y_val)), y_val, alpha=0.3)
        
        plt.title('GA Learning Rate Optimizer')
        plt.legend(['y_pred', 'y_val'])
        plt.show()