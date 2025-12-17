from .scikit_learn_testing_template import test_scikit_learn_baseline
from src.machine_learn.imports import MLPRegressor

def test_scikit_learn_non_linear():
    optimizer = MLPRegressor()

    test_scikit_learn_baseline(optimizer, training_args={
        'maximum iterations' : 1000,
        'hidden_layer_sizes' : (100, 100, 100),
        'activation' : 'relu'})