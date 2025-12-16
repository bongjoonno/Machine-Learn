from src.machine_learn.models import LinearRegression
from .linear_regression_testing_template import linear_regression_test_template

def test_linear_regression() -> None:
    optimizer = LinearRegression()
    linear_regression_test_template(optimizer, {}, optimize_lr=True, early_stop=True)