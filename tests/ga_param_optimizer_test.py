from src.machine_learn.genetic_algorithms import GAOptimizer
from .model_testing_template import model_test_template

def test_ga_param_optimizer() -> None:
        ga_optimizer = GAOptimizer()
        model_test_template(ga_optimizer, {'non_linearity' : True}, early_stop=True, scale_y=True)