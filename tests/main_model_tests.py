from tests.linear_regression_test import linear_regression_test
from tests.logistic_regression_test import logistic_regression_test
from tests.naive_bayes_test import naive_bayes_test
from src.machine_learn.imports import np

tests = [linear_regression_test, logistic_regression_test, naive_bayes_test]

def main_model_tests() -> list[float, float, float]:
    return [func() for func in tests]