from src.machine_learn.imports import np, random
from src.machine_learn.constants import EPOCHS
from src.machine_learn.types import DF, Series, NDArray
from src.machine_learn.metrics import mean_squared_error, r_squared
from src.machine_learn.genetic_algorithms import GeneticAlgorithm
import inspect
# still need to determine how to determine bounds

param_lower_bound = -0.8568
param_upper_bound = abs(param_lower_bound)

sigma_for_mutation = 0.0001
population_size = 1000

non_linear_functions = [np.sin, np.tanh, lambda x: x, np.square,]

class GANONLinearOptimizer:
    min_delta = 0.0001
    patience = 5
    
    def train(self, 
              x_train: DF, 
              y_train: Series, 
              x_val: DF | None = None, 
              y_val: Series | None = None, 
              epochs: int | None = None, 
              mutate: bool = False, 
              non_linearity: bool = False) -> None:  
        
        if epochs is not None:
            early_stop = False
        elif x_val is not None and y_val is not None and epochs is None:
            X_val = np.column_stack((np.ones(len(x_val)), x_val))
            self.min_val_mse = float('inf')
            early_stop = True
        else:
            epochs = EPOCHS
        
        X = np.column_stack((np.ones(len(x_train)), x_train))
        
        self.min_train_mse = float('inf')
        
        number_of_features = X.shape[1]

        if non_linearity:
            functions = [[np.random.choice(non_linear_functions) for _ in range(number_of_features)] for _ in range(population_size)]
 
        population = [np.random.uniform(param_lower_bound, param_upper_bound, number_of_features) for _ in range(population_size)]
        
        losses = [0 for _ in range(population_size)]
        
        self.epochs_performed = 0
        no_improvement = 0
        
        while True:
            self.epochs_performed += 1
            
            for i, solution in enumerate(population):
                if non_linearity:
                    y_pred = X * solution
                    y_pred = np.sum(np.column_stack([f(X[:, j]) for j, f in enumerate(functions[i])]), axis=1)
                else:
                    y_pred = X @ solution
                    
                losses[i] = mean_squared_error(y_pred, y_train)
            
            train_generation_min_mse = min(losses)
            
            self.min_train_mse = min(train_generation_min_mse, self.min_train_mse)
            
            if early_stop:
                for i, solution in enumerate(population):
                    if non_linearity:
                        y_pred = X_val * solution
                        y_pred = np.sum(np.column_stack([f(X_val[:, j]) for j, f in enumerate(functions[i])]), axis=1)
                    else:
                        y_pred = X_val @ solution
                        
                    losses[i] = mean_squared_error(y_pred, y_val)
            
                val_generation_min_mse = min(losses)

                if self.min_val_mse - val_generation_min_mse < GANONLinearOptimizer.min_delta:
                    no_improvement += 1
                    
                    if no_improvement >= GANONLinearOptimizer.patience:
                        break
                else:
                    self.min_val_mse = val_generation_min_mse
                    no_improvement = 0
            
            elif self.epochs_performed == epochs:
                break
                
            top_50_percent_of_population = [solution for _, solution in sorted(zip(losses, population), key=lambda x: x[0])][:population_size//2]
            
            children = []
            
            for i in range(number_of_features):
                params = []
                    
                for j in range(len(top_50_percent_of_population)):
                    params.append(top_50_percent_of_population[j][i])
                
                param_children = GeneticAlgorithm.make_offspring(params)
                
                if mutate:
                    for k in range(len(param_children)):
                        if np.random.random() < 0.01:
                            param_children[k] += random.gauss(0, sigma = sigma_for_mutation)
                
                children.append(param_children)
                        
                    
            
            children = np.array(children).T
            children = children.tolist()
            
            population = top_50_percent_of_population + children
            
            if non_linearity:
                top_50_percent_of_functions = [solution for _, solution in sorted(zip(losses, functions), key=lambda x: x[0])][:population_size//2]
                functions = top_50_percent_of_functions + top_50_percent_of_functions
        
                #for func in functions:
                    #print(func)
                    
        self.theta = population[0]
    
    def predict(self, x: DF) -> NDArray:
        X = np.column_stack((np.ones(len(x)), x))
        y = X @ self.theta
        return y