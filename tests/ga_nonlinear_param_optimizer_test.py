from src.machine_learn.genetic_algorithms import GANONLinearOptimizer
from .model_testing_template import model_test_template

def test_nonlinear_ga_param_optimizer() -> None:
        ga_optimizer = GANONLinearOptimizer()
        return model_test_template(ga_optimizer, training_args={'non_linearity' : True, 'crossover_method' : 'sbx'}, early_stop=True, scale_y=True)