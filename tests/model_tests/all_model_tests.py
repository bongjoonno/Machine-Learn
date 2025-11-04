from model_tests.linear_regression_testing import linear_regression_test
from model_tests.logistic_regression_test import logistic_regression_test
from model_tests.naive_bayes_testing import naive_bayes_test

def run_all_tests():
    return linear_regression_test(), logistic_regression_test(), naive_bayes_test()