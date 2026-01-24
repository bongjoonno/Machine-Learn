from src.machine_learn.imports import np

def mean_absolute_error(y_pred, y_act):
    return float(np.mean(abs(y_pred-y_act)))