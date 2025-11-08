from src.machine_learn.imports import pd
from src.machine_learn.types import DF

def train_test_split(x: DF, y: DF, train_size: float = 0.8, test_size: float = 0.2) -> tuple[DF, DF, DF, DF]:
    if train_size + test_size != 1:
        raise ValueError('Please enter train_size and test_size that sum to 1')
    
    n = len(x)
    
    training_border = int(train_size * n)
    
    x_train = x[:training_border]
    y_train = y[:training_border]
    
    x_test = x[training_border:]
    y_test = y[training_border:]
    
    return x_train, y_train, x_test, y_test