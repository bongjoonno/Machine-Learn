from src.machine_learn.imports import np
from src.machine_learn.types import DF, Series, NDArray
from src.machine_learn.constants import EPOCHS, LEARNING_RATE

class LogisticRegression:
    def train(self, x: DF | Series, y: Series, epochs: int = EPOCHS, learning_rate: float = LEARNING_RATE) -> None:
        X = np.column_stack((np.ones(len(x)), x))
        one_divided_by_n = 1/X.shape[0]
        
        self.theta = np.zeros(X.shape[1])
        self.min_loss = float('inf')
        
        for _ in range(epochs):
            y_pred = LogisticRegression.sigmoid(X @ self.theta)
            errors = y_pred - y
            gradient = one_divided_by_n * (X.T @ errors)
            self.theta -= learning_rate * gradient
            
            # needs to be fixed, mse is not the way to measure loss for binary classification
            mse = np.mean(errors**2)
            self.min_loss = min(self.min_loss, mse)
        
    @staticmethod
    def sigmoid(x: NDArray) -> NDArray:
        x = np.clip(x, -709, 709)
        return np.where(x >= 0, 
                    1 / (1 + np.exp(-x)), 
                    np.exp(x) / (1 + np.exp(x)))
    
    def predict(self, x: DF | Series) -> NDArray:
        X = np.column_stack((np.ones(len(x)), x))
        
        y = LogisticRegression.sigmoid(X @ self.theta)
        y_categorical = (y >= 0.5).astype(int)
        
        return y_categorical