from src.machine_learn.data_prep import regression_test_data
from src.machine_learn.imports import np, plt, tqdm, tqdm_joblib, StandardScaler, Parallel, delayed
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
    
    n = 1
    
    avg_r2s = []
    
    all_data_folds = [split_k_folds(x, y, n) for x, y, _ in regression_test_data]
    all_cols_to_scale = [cols_to_scale for _, _, cols_to_scale in regression_test_data]
    
    training_packages = [(fold, cols_to_scale, optimizer, training_args, early_stop, optimize_lr, scale_y) 
                         for fold, cols_to_scale in zip(all_data_folds, all_cols_to_scale)]
        
        
    parallel_obj = Parallel(n_jobs=6)
    
    
    avg_r2s = [k_cross_validation_train(*args) for args in training_packages]
    
    return avg_r2s
        
        

def k_cross_validation_train(*args):
    folds, cols_to_scale, optimizer, training_args, early_stop, optimize_lr, scale_y = args

    args_packages = [(fold, cols_to_scale, optimizer, training_args, early_stop, optimize_lr, scale_y)
                     for fold in folds]
    
   
    r2s = [fold_train(*args) for args in args_packages]
    print(r2s)
    return np.mean(r2s)

    
    
def fold_train(*args):
    fold, cols_to_scale, optimizer, training_args, early_stop, optimize_lr, scale_y = args
    
    x_train, y_train, x_val, y_val = fold
    
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
    
    return r_squared(y_pred, y_val)

    