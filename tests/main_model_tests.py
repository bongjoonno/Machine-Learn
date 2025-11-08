from . import linear_regression_test, logistic_regression_test, naive_bayes_test
from src.machine_learn.imports import np

tests = [linear_regression_test, logistic_regression_test, naive_bayes_test]

def main_model_tests() -> list[np.float64, float, float]:
    return [func() for func in tests]