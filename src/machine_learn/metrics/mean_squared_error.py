from src.machine_learn.imports import cp

def mean_squared_error(y_pred, y_act):
    return float(cp.mean((y_pred-y_act)**2))