from . import test_linear_regression, test_logistic_regression, test_naive_bayes
from src.machine_learn.imports import np

tests = [test_linear_regression, test_logistic_regression, test_naive_bayes]

def main_model_tests() -> list[float, float, float]:
    return [func() for func in tests]