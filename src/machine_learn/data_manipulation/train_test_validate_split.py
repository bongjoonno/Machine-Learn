from src.machine_learn.types import DF, Series, NDArray

def train_test_validate_split(x: DF, y: Series, train_size: float = 0.80, validation_size: float = 0.10, test_size: float = 0.10) -> tuple[DF, Series, DF, Series, DF, Series]:
    if sum([train_size, test_size, validation_size]) != 1:
        raise ValueError('train_size, test_size, and validation_size must sum to 1')
   
    n = len(x)
    
    train_border = int(train_size * n)
    validation_border = train_border + int(validation_size * n)

    
    x_train = x[:train_border]
    y_train = y[:train_border]
    
    x_validate = x[train_border : validation_border]
    y_validate = y[train_border : validation_border]
    
    x_test = x[validation_border:]
    y_test = y[validation_border:]

    
    return x_train, y_train, x_validate, y_validate, x_test, y_test