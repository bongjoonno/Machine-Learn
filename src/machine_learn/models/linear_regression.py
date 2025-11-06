from src.machine_learn.imports import np
from src.machine_learn.constants import EPOCHS, LEARNING_RATE

class LinearRegression:
    def train(self, x, y, epochs=EPOCHS, learning_rate=LEARNING_RATE):   
        self.x = np.column_stack((np.ones(len(x)), x))
        self.y = y
        self.one_divided_by_n = 1/(len(self.x))

        self.theta = np.zeros(self.x.shape[1])
        self.min_loss = float('inf')

        for _ in range(epochs):
            y_pred = self.x @ self.theta
            errors = y_pred - self.y
            gradient = self.one_divided_by_n * (self.x.T @ errors)
            self.theta -= learning_rate * gradient

            mse = np.mean(errors**2)
            self.min_loss = min(mse, self.min_loss)
            print(self.min_loss)

    
    def predict(self, x_test):
        x_test = np.column_stack((np.ones(len(x_test)), x_test))
        y_pred = x_test @ self.theta
        return y_pred