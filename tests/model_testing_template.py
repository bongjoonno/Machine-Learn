from src.machine_learn.data_prep import linear_regression_test_data
from src.machine_learn.imports import plt, StandardScaler
from src.machine_learn.metrics import r_squared
from src.machine_learn.data_manipulation import train_test_split, scale_data
from src.machine_learn.models import LinearRegression
from src.machine_learn.genetic_algorithms import GAOptimizer, GAlrOptimizer

def model_test_template(optimizer: LinearRegression | GAOptimizer, training_args: dict = {}, early_stop: bool = True, optimize_lr: bool = False, scale_y: bool = False):
    scaler = StandardScaler()
        
        
    for x, y, cols_to_scale in linear_regression_test_data:
        x_train, y_train, x_val, y_val = train_test_split(x, y)
        x_train, x_val = scale_data(x_train, x_val, columns_to_scale=cols_to_scale)
            
        if scale_y:
            y_train = scaler.fit_transform(y_train.to_numpy().reshape(-1, 1)).flatten()
            y_val = scaler.transform(y_val.to_numpy().reshape(-1, 1)).flatten()
        
        if optimize_lr:
            lr_optimizer = GAlrOptimizer(optimizer, x_train, y_train, x_val, y_val)
            lr = lr_optimizer.optimize_lr()
            training_args['learning_rate'] = lr
            
        if early_stop:
            optimizer.train(x_train, y_train, x_val, y_val, **training_args)
        else:
            optimizer.train(x_train, y_train, **training_args)

        y_pred = optimizer.predict(x_val)
        
        r2 = r_squared(y_pred, y_val)
        print(f'{r2=}')
        
        plt.plot(range(len(y_val)), y_pred)
        plt.plot(range(len(y_val)), y_val)
        plt.legend(['y_pred', 'y_val'])
        plt.show()