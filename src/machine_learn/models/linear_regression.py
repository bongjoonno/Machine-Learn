from src.machine_learn.imports import np
from src.machine_learn.types import DF, Series, NDArray
from src.machine_learn.constants import EPOCHS, LEARNING_RATE
from src.machine_learn.metrics import mean_squared_error

class LinearRegression:
    def train(self, x_train: DF, y_train: Series, epochs: int = EPOCHS, learning_rate: float = LEARNING_RATE) -> None:   
        X = np.column_stack((np.ones(len(x_train)), x_train))
        one_divided_by_n = 1/(X.shape[1])

        self.theta = np.ones(X.shape[1])
        self.min_train_loss = float('inf')

        for _ in range(epochs):
            y_pred = X @ self.theta
            errors = y_pred - y_train
            gradient = one_divided_by_n * (X.T @ errors)
            self.theta -= learning_rate * gradient

            train_mse = mean_squared_error(y_pred, y_train)
            self.min_train_loss = min(train_mse, self.min_train_loss)
    
    def train_early_stop(self, x_train: DF, y_train: Series, x_val: DF, y_val: Series, learning_rate: float = LEARNING_RATE) -> None:   
        X = np.column_stack((np.ones(len(x_train)), x_train))
        X_val = np.column_stack((np.ones(len(x_val)), x_val))
        
        one_divided_by_n = 1/(X.shape[1])

        self.theta = np.ones(X.shape[1])
        self.min_train_mse = float('inf')
        self.min_val_mse = float('inf')
        
        self.epochs = 0
        
        while True:
            self.epochs += 1
            
            y_pred = X @ self.theta
            val_y_pred = X_val @ self.theta
            
            errors = y_pred - y_train
            gradient = one_divided_by_n * (X.T @ errors)
            self.theta -= learning_rate * gradient

            train_mse = mean_squared_error(y_pred, y_train)
            self.min_train_mse = min(train_mse, self.min_train_mse)
            
            val_mse = mean_squared_error(val_y_pred, y_val)
            
            if val_mse >= self.min_val_mse:
                break
            else:
                self.min_val_mse = val_mse
            
            
            
            

    
    def predict(self, x: DF) -> NDArray:
        X = np.column_stack((np.ones(len(x)), x))
        y = X @ self.theta
        return y