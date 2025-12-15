from src.machine_learn.imports import np
from src.machine_learn.types import DF, Series, NDArray
from src.machine_learn.constants import EPOCHS, LEARNING_RATE
from src.machine_learn.metrics import mean_squared_error

class LinearRegression:
    min_delta = 0.0001
    patience = 5
    
    def __init__(self):
        self.min_val_mse = float('inf')
        
    def train(self, x_train: DF, y_train: Series, x_val: DF | None = None, y_val: Series | None = None, epochs: int | None = None, learning_rate: float = LEARNING_RATE) -> None:  
        early_stop = False
        
        if epochs is None:
            if x_val is not None and y_val is not None:
                X_val = np.column_stack((np.ones(len(x_val)), x_val))
                early_stop = True
                
                
            else:
                epochs = EPOCHS
            
        X = np.column_stack((np.ones(len(x_train)), x_train))
        
        one_divided_by_n = 1/(X.shape[0])

        self.theta = np.ones(X.shape[1])
        self.min_train_mse = float('inf')
        
        self.epochs_performed = 0
        no_improvement = 0
        
        while True:
            self.epochs_performed += 1
            
            y_pred = X @ self.theta
            
            errors = y_pred - y_train
            gradient = one_divided_by_n * (X.T @ errors)
            self.theta -= learning_rate * gradient

            train_mse = mean_squared_error(y_pred, y_train)
            self.min_train_mse = min(train_mse, self.min_train_mse)
            
            if early_stop:
                val_y_pred = X_val @ self.theta
                val_mse = mean_squared_error(val_y_pred, y_val)
                
                if self.min_val_mse - val_mse < LinearRegression.min_delta:
                    no_improvement += 1
                    if no_improvement == LinearRegression.patience:
                        break
                else:
                    self.min_val_mse = val_mse
                    no_improvement = 0
            
            elif self.epochs_performed == epochs:
                break
                
            
    
    def predict(self, x: DF) -> NDArray:
        X = np.column_stack((np.ones(len(x)), x))
        y = X @ self.theta
        return y