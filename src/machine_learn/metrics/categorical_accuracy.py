from src.machine_learn.imports import np

def categorical_accuracy(y_pred: list[int], y_test):
    return sum(y_pred == y_test) / len(y_pred)