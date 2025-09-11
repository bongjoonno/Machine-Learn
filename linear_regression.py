from imports import np

class LinearRegression:
    def __init__(self, x_train, y_train, x_test, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.theta = np.random.rand(self.x_train.shape[1])

    def train(self, epochs=1000, learning_rate=0.05):
        n = len(self.x_train)
        
        self.x_train = np.column_stack((np.ones(len(self.x_train)), self.x_train))
        self.x_test = np.column_stack((np.ones(len(self.x_test)), self.x_test))
        
        one_divided_by_n = 1/n
        
        for _ in range(epochs):
            y_pred = self.x_train @ self.theta
            errors = y_pred - self.y_train
            gradient = one_divided_by_n * (self.x_train.T @ errors)
            self.theta -= learning_rate * gradient
    
    def test(self):
        y_pred = self.x_test @ self.theta
        return sum(y_pred == self.y_test)
            
        


#LinearRegression() Object
#train and test methods