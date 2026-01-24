from src.machine_learn.data_prep import regression_test_data
from src.machine_learn.imports import np, StandardScaler
from src.machine_learn.metrics import r_squared
from src.machine_learn.data_manipulation import scale_data, split_k_folds
from src.machine_learn.models import LinearRegression
from src.machine_learn.genetic_algorithms import GAOptimizer, GAlrOptimizer, GANONLinearOptimizer

scaler = StandardScaler()

def model_test_template(optimizer: LinearRegression | GAOptimizer | GANONLinearOptimizer, 
                        training_args: dict = {}, 
                        early_stop: bool = True, 
                        optimize_lr: bool = False, 
                        scale_y: bool = False):
    
    r2s = []

    for data in regression_test_data:
        x_train, y_train, x_val, y_val = data*
        
        if optimize_lr:
            lr_optimizer = GAlrOptimizer(optimizer, x_train, y_train, x_val, y_val)
            lr = lr_optimizer.optimize_lr()
            training_args['learning_rate'] = lr
            
        if early_stop:
            optimizer.train(x_train, y_train, x_val, y_val, **training_args)
        else:
            optimizer.train(x_train, y_train, **training_args)

        y_pred = optimizer.predict(x_val)
        
        r2s.append(r_squared(y_pred, y_val))
    
    return r2s