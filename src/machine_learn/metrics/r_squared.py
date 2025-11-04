from src.machine_learn.imports import np

def r_squared(y_pred, y_test):
    sum_squared_prediction_difference = np.sum((y_test - y_pred)**2)
    sum_squared_mean_difference = np.sum((y_test - np.mean(y_test))**2)
        
    R_squared = 1 - (sum_squared_prediction_difference / sum_squared_mean_difference)
        
    return R_squared