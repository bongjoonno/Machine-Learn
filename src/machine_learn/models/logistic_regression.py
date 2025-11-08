from src.machine_learn.imports import np
from src.machine_learn.types import DF, NDArray
from src.machine_learn.constants import EPOCHS, LEARNING_RATE

class LogisticRegression:
    def train(self, x: DF, y: DF, epochs: int = EPOCHS, learning_rate: float = LEARNING_RATE) -> None:
        x = np.column_stack((np.ones(len(x)), x))
        one_divided_by_n = 1/len(x)
        
        self.theta = np.zeros(x.shape[1])
        self.min_loss = float('inf')
        
        for _ in range(epochs):
            y_pred = LogisticRegression.sigmoid(x @ self.theta)
            errors = y_pred - y
            gradient = one_divided_by_n * (x.T @ errors)
            self.theta -= learning_rate * gradient
            
            mse = np.mean(errors**2)
            self.min_loss = min(self.min_loss, mse)
        
    @staticmethod
    def sigmoid(x: NDArray) -> NDArray:
        x = np.clip(x, -709, 709)
        return np.where(x >= 0, 
                    1 / (1 + np.exp(-x)), 
                    np.exp(x) / (1 + np.exp(x)))
    
    def predict(self, x_test: DF) -> list[int]:
        x_test = np.column_stack((np.ones(len(x_test)), x_test))
        
        y_pred = LogisticRegression.sigmoid(x_test @ self.theta)
        y_pred_categorical = (y_pred >= 0.5).astype(int)
        
        return y_pred_categorical