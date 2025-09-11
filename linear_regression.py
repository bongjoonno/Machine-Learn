from imports import np

class LinearRegression:
    def __init__(self):
        pass

    def train(self, x_train, y_train, epochs=1000, learning_rate=0.05):
        self.x_train = x_train
        self.y_train = y_train

        n = len(self.x_train)
        
        self.x_train = np.column_stack((np.ones(len(self.x_train)), self.x_train))
        
        one_divided_by_n = 1/n
        
        self.theta = np.random.rand(self.x_train.shape[1])
        
        for _ in range(epochs):
            y_pred = self.x_train @ self.theta
            errors = y_pred - self.y_train
            gradient = one_divided_by_n * (self.x_train.T @ errors)
            self.theta -= learning_rate * gradient
    
    def predict(self, x):
        x = np.column_stack((np.ones(len(x)), x))
        y_pred = x @ self.theta
        return y_pred