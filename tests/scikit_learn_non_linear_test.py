from .scikit_learn_testing_template import test_scikit_learn_baseline
from src.machine_learn.imports import MLPRegressor

def test_scikit_learn_non_linear():
    optimizer = MLPRegressor(max_iter=100_000)

    test_scikit_learn_baseline(optimizer, linear_data = False)