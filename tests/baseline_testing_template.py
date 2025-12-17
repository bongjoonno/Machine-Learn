from src.machine_learn.imports import StandardScaler, LinearRegression, MLPRegressor, TabPFNRegressor, plt
from src.machine_learn.data_prep import linear_regression_test_data, non_linear_test_data
from src.machine_learn.data_manipulation import train_test_split, scale_data
from src.machine_learn.metrics import r_squared

def test_baseline_models(optimizer: LinearRegression | MLPRegressor | TabPFNRegressor,
                               linear_data: bool = True,
                               scale_y: bool = False):
    scaler = StandardScaler()
    
    test_data = linear_regression_test_data if linear_data else non_linear_test_data
        
    for x, y, cols_to_scale in test_data:
        x_train, y_train, x_val, y_val = train_test_split(x, y)
        x_train, x_val = scale_data(x_train, x_val, columns_to_scale=cols_to_scale)
            
        if scale_y:
            y_train = scaler.fit_transform(y_train.to_numpy().reshape(-1, 1)).flatten()
            y_val = scaler.transform(y_val.to_numpy().reshape(-1, 1)).flatten()

        optimizer.fit(x_train, y_train)
        
        y_pred = optimizer.predict(x_val)
        
        r2 = r_squared(y_pred, y_val)
        print(f'{r2=}')
        
        plt.plot(range(len(y_val)), y_pred)
        plt.plot(range(len(y_val)), y_val)
        plt.legend(['y_pred', 'y_val'])
        plt.show()