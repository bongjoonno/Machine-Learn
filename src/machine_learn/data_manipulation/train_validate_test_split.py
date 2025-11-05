def train_validate_test_split(x, y, train_size=0.60, test_size=0.35):
    n = len(x)
    
    train_border = int(train_size * n)
    test_border = train_border + int(test_size * n)

    
    x_train = x[:train_border]
    y_train = y[:train_border]
    
    x_test = x[train_border : test_border]
    y_test = x[train_border : test_border]

    x_validate = x[test_border:]
    y_validate = y[test_border:]
    
    return x_train, y_train, x_test, y_test, x_validate, y_validate