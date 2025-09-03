from imports import np

def linear_regression(x_train, y_train, x_test, y_test, epochs=1000, learning_rate=0.05):
    
    x_train = np.column_stack((np.ones(len(x_train)), x_train))
    x_test = np.column_stack((np.ones(len(x_test)), x_test))
    
    theta = np.random.rand(x_train.shape[1])
    
    for _ in range(epochs):
        y_pred = x_train @ theta
        errors = y_pred - y_train
        gradient = (1/n) * (errors @ x_train)
        theta -= learning_rate * gradient
        
    y_pred_test = x_test @ theta
    
    return y_pred_test

res = linear_regression([1,2,3,4,5,6,7,8,9,10], [1,2,3,4,5,6,7,8,9,10])
print(res)  