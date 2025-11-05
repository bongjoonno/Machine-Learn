from src.machine_learn.imports import np
from src.machine_learn.constants import EPOCHS, LEARNING_RATE

class LinearRegression:
    def __init__(self, x_train, y_train):
        self.x_train = np.column_stack((np.ones(len(x_train)), x_train))
        self.y_train = y_train
        self.one_divided_by_n = 1/(len(self.x_train))

    def train(self, epochs=EPOCHS, learning_rate=LEARNING_RATE):        
        self.theta = np.zeros(self.x_train.shape[1])
        
        for _ in range(epochs):
            y_pred = self.x_train @ self.theta
            errors = y_pred - self.y_train
            gradient = self.one_divided_by_n * (self.x_train.T @ errors)
            self.theta -= learning_rate * gradient
    
    def predict(self, x_test):
        x_test = np.column_stack((np.ones(len(x_test)), x_test))
        y_pred = x_test @ self.theta
        return y_pred