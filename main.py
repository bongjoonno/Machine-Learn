from model_tests.linear_regression_testing import linear_regression_test
from model_tests.logistic_regression_test import logistic_regression_test
from model_tests.naive_bayes_testing import naive_bayes_test

if __name__ == '__main__':
    print(linear_regression_test())
    print(logistic_regression_test())
    print(naive_bayes_test())