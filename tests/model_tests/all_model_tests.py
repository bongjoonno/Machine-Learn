from tests.model_tests.linear_regression_test import linear_regression_test
from tests.model_tests.logistic_regression_test import logistic_regression_test
from tests.model_tests.naive_bayes_test import naive_bayes_test

def run_all_tests():
    return linear_regression_test(), logistic_regression_test(), naive_bayes_test()