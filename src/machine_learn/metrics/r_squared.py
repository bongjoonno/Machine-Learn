from src.machine_learn.imports import np
from src.machine_learn.types import NDArray, Series

def r_squared(y_pred: NDArray, y_test: Series) -> float:
    sum_squared_prediction_difference = np.sum((y_test - y_pred)**2)
    sum_squared_mean_difference = np.sum((y_test - np.mean(y_test))**2)
        
    r2 = 1 - (sum_squared_prediction_difference / sum_squared_mean_difference)
        
    return float(r2)