from src.machine_learn.imports import np
from src.machine_learn.types import DF, Series, NDArray
from src.machine_learn.constants import EPOCHS, LEARNING_RATE
from src.machine_learn.metrics import mean_squared_error

class LinearRegression:
    def train(self, x: DF, y: Series, epochs: int = EPOCHS, learning_rate: float = LEARNING_RATE) -> None:   
        X = np.column_stack((np.ones(len(x)), x))
        one_divided_by_n = 1/(X.shape[1])

        self.theta = np.ones(X.shape[1])
        self.min_loss = float('inf')

        for _ in range(epochs):
            y_pred = X @ self.theta
            errors = y_pred - y
            gradient = one_divided_by_n * (X.T @ errors)
            self.theta -= learning_rate * gradient

            mse = mean_squared_error(y_pred, y)
            self.min_loss = min(mse, self.min_loss)

    
    def predict(self, x: DF) -> NDArray:
        X = np.column_stack((np.ones(len(x)), x))
        y = X @ self.theta
        return y