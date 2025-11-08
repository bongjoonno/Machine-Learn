from src.machine_learn.types import DF, Series

def train_test_split(x: DF, y: DF, train_size: float = 0.8, test_size: float = 0.2) -> tuple[DF, Series, DF, Series]:
    if sum([train_size, test_size]) != 1:
        raise ValueError('train_size and test_size must sum to 1')
    
    n = len(x)
    
    training_border = int(train_size * n)
    
    x_train = x[:training_border]
    y_train = y[:training_border]
    
    x_test = x[training_border:]
    y_test = y[training_border:]
    
    return x_train, y_train, x_test, y_test