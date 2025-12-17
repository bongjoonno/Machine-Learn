from .baseline_testing_template import test_baseline_models
from src.machine_learn.imports import TabPFNRegressor

model_path = r'C:\main\code\repos\Machine-Learn\baseline_models\tabpfn-v2.5-regressor-v2.5_default.ckpt'

optimizer = TabPFNRegressor(model_path=model_path, device="cpu", ignore_pretraining_limits=True) 

def test_tab_pfn_non_linear():
    test_baseline_models(optimizer, data_type='all')