from . import test_linear_regression, test_logistic_regression, test_naive_bayes

tests = [test_linear_regression, test_logistic_regression, test_naive_bayes]

def test_all_models() -> list[float]:
    return [func() for func in tests]