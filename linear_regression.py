import numpy as np

def linear_regression(x, y, epochs=1000, learning_rate=0.05, training_split=0.8):
    x_train = x[:training_split]
    y_train = y[:training_split]
    
    x_test = x[training_split:]
    y_test = y[training_split:]
    
    x_train = np.column_stack((np.ones(len(x_train)), x_train))
    x_test = np.column_stack((np.ones(len(x_test)), x_test))
    
    theta = np.random.rand(x_train.shape[1])
    
    return theta, len(x_train) + len(x_test) == len(x)

res = linear_regression([1,2,3,4], [1,2,3,4])
print(res)
        