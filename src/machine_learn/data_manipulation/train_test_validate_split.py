from src.machine_learn.types import DF, Series, NDArray

def train_test_validate_split(x: DF, y: Series, train_size: float = 0.80, test_size: float = 0.10, validation_size: float = 0.10) -> tuple[NDArray, NDArray, NDArray, NDArray, NDArray, NDArray]:
    if sum([train_size, test_size, validation_size]) != 1:
        raise ValueError('train_size, test_size, and validation_size must sum to 1')
   
    n = len(x)
    
    train_border = int(train_size * n)
    test_border = train_border + int(test_size * n)

    
    x_train = x[:train_border].to_numpy().flatten()
    y_train = y[:train_border].to_numpy()
    
    x_test = x[train_border : test_border].to_numpy().flatten()
    y_test = y[train_border : test_border].to_numpy()

    x_validate = x[test_border:].to_numpy().flatten()
    y_validate = y[test_border:].to_numpy()
    
    return x_train, y_train, x_test, y_test, x_validate, y_validate