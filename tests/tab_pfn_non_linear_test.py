from .baseline_testing_template import test_baseline_models
from src.machine_learn.imports import TabPFNRegressor

def test_tab_pfn_non_linear():
    optimizer = TabPFNRegressor()

    test_baseline_models(optimizer, linear_data = False)