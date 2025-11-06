def train_test_split(x, y, train_size=0.8):
    n = len(x)
    
    training_border = int(train_size * n)
    
    x_train = x[:training_border]
    y_train = y[:training_border]
    
    x_test = x[training_border:]
    y_test = y[training_border:]
    
    return x_train, y_train, x_test, y_test