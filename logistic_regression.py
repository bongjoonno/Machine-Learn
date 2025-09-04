from imports import np

def logistic_regression(x_train, y_train, x_test, epochs=1000, learning_rate=0.05):
    n = len(x_train)
    
    x_train = np.column_stack((np.ones(len(x_train)), x_train))
    x_test = np.column_stack((np.ones(len(x_test)), x_test))
    
    theta = np.random.rand(x_train.shape[1])
    
    one_divided_by_n = 1/n
    
    for _ in range(epochs):
        y_pred = sigmoid(x_train @ theta)
        errors = y_pred - y_train
        gradient = one_divided_by_n * (x_train.T @ errors)
        theta -= learning_rate * gradient
        
    y_pred_test = sigmoid(x_test @ theta)
    
    y_pred_test = (y_pred_test >= 0.5).astype(int)
    return y_pred_test


def sigmoid(x):
    return np.where(x >= 0, 
                   1 / (1 + np.exp(-x)), 
                   np.exp(x) / (1 + np.exp(x)))