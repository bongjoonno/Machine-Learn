from src.machine_learn.imports import np, pd, StandardScaler, LinearRegression, TabPFNRegressor, plt
from src.machine_learn.data_prep import baseline_model_regression_test_data
from src.machine_learn.data_manipulation import train_test_split, scale_data
from src.machine_learn.metrics import r_squared

def test_baseline_models(optimizer: LinearRegression | TabPFNRegressor, scale_y: bool = False):
    scaler = StandardScaler()
    
    averaged_r2s = []
    n = 10
   
    for x, y, cols_to_scale in baseline_model_regression_test_data:
        test_size = int(len(x)*(1/n))
        
        r2s = []

        for i in range(0, len(x)-test_size, test_size):
            x_train = pd.concat([x[:i], x[i+test_size:]])
            x_val = x[i: i+test_size]
            
            y_train = pd.concat([y[:i], y[i+test_size:]])
            y_val = y[i: i+test_size]
            x_train, x_val = scale_data(x_train, x_val, columns_to_scale=cols_to_scale)
                
            if scale_y:
                y_train = scaler.fit_transform(y_train.to_numpy().reshape(-1, 1)).flatten()
                y_val = scaler.transform(y_val.to_numpy().reshape(-1, 1)).flatten()

            optimizer.fit(x_train, y_train)
            
            y_pred = optimizer.predict(x_val)
            
            r2s.append(r_squared(y_pred, y_val))
        averaged_r2s.append(np.mean(r2s))
    
    return averaged_r2s 