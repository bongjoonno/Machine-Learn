from src.machine_learn.genetic_algorithms import GAOptimizer
from .linear_regression_testing_template import linear_regression_test_template

def test_ga_param_optimizer() -> None:
        ga_optimizer = GAOptimizer()
        linear_regression_test_template(ga_optimizer, {'non_linearity' : True}, early_stop=True, scale_y=True)