from src.machine_learn.genetic_algorithms import GANONLinearOptimizer
from .model_testing_template import model_test_template

def test_nonlinear_ga_param_optimizer() -> None:
        ga_optimizer = GANONLinearOptimizer()
        model_test_template(ga_optimizer, {'non_linearity' : True}, early_stop=True, scale_y=True, linear_data=True)