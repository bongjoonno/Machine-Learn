from src.machine_learn.models import LinearRegression
from .model_testing_template import model_test_template

def test_ga_lr_optimizer() -> None:
    optimizer = LinearRegression()
    model_test_template(optimizer, {}, optimize_lr=True, early_stop=True)