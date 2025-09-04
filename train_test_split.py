from imports import np

def train_test_split(x, y, train_split=0.8):
    x = np.array(x)
    y = np.array(y)
    
    n = len(x)
    
    indices = np.random.permutation(n) 
    
    x = x[indices]
    y = y[indices]
    
    training_border = int(train_split * n)
    
    x_train = x[:training_border]
    y_train = y[:training_border]
    
    x_test = x[training_border:]
    y_test = y[training_border:]
    
    return x_train, y_train, x_test, y_test