from src.machine_learn.genetic_algorithms import GAOptimizer
from .linear_regression_testing_template import linear_regression_test_template

def test_ga_all_param_optimizer():
        ga_optimizer = GAOptimizer()
        linear_regression_test_template(ga_optimizer, (), early_stop=True, scale_y=True)