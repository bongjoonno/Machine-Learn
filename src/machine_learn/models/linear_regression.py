from src.machine_learn.imports import np
from src.machine_learn.types import DF, NDArray
from src.machine_learn.constants import EPOCHS, LEARNING_RATE

class LinearRegression:
    def train(self, x: DF, y: DF, epochs: int = EPOCHS, learning_rate: float = LEARNING_RATE) -> None:   
        x = np.column_stack((np.ones(len(x)), x))
        one_divided_by_n = 1/(x.shape[1])

        self.theta = np.ones(x.shape[1])
        self.min_loss = float('inf')

        for _ in range(epochs):
            y_pred = x @ self.theta
            errors = y_pred - y
            gradient = one_divided_by_n * (x.T @ errors)
            self.theta -= learning_rate * gradient

            mse = np.mean(errors**2)
            self.min_loss = min(mse, self.min_loss)

    
    def predict(self, x_test: DF) -> NDArray:
        x_test = np.column_stack((np.ones(len(x_test)), x_test))
        y_pred = x_test @ self.theta
        return y_pred