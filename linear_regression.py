from imports import np

def linear_regression(x_train, y_train, x_test, epochs=1000, learning_rate=0.05):
    n = len(x_train)
    
    x_train = np.column_stack((np.ones(len(x_train)), x_train))
    x_test = np.column_stack((np.ones(len(x_test)), x_test))
    
    theta = np.random.rand(x_train.shape[1])
    
    one_divided_by_n = 1/n
    
    for _ in range(epochs):
        y_pred = x_train @ theta
        errors = y_pred - y_train
        gradient = one_divided_by_n * (x_train.T @ errors)
        theta -= learning_rate * gradient
        
    y_pred_test = x_test @ theta
    
    return y_pred_test