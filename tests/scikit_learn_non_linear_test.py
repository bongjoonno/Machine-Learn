from .scikit_learn_testing_template import test_scikit_learn_baseline
from src.machine_learn.imports import MLPRegressor

def test_scikit_learn_non_linear():
    optimizer = MLPRegressor(hidden_layer_sizes=(100, 100), activation='relu', max_iter=1000)

    test_scikit_learn_baseline(optimizer, linear_data = False)