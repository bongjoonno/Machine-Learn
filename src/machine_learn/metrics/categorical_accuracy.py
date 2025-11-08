from src.machine_learn.imports import np
from src.machine_learn.types import NDArray, Series

def categorical_accuracy(y_pred: NDArray, y_test: Series) -> float:
    return sum(y_pred == y_test) / len(y_pred)