from src.machine_learn.imports import np

def mse(y_pred, y_act):
    return np.mean((y_pred-y_act)**2)