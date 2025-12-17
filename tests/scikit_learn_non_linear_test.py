from .baseline_testing_template import test_baseline_models
from src.machine_learn.imports import MLPRegressor

def test_scikit_learn_non_linear():
    optimizer = MLPRegressor(max_iter=100_000)

    test_baseline_models(optimizer, linear_data = False)