from src.machine_learn.imports import np, random, sp
from src.machine_learn.constants import EPOCHS, X_VARIABLE
from src.machine_learn.types import DF, Series, NDArray
from src.machine_learn.metrics import mean_squared_error
from src.machine_learn.genetic_algorithms import GeneticAlgorithm

param_lower_bound = -0.8568
param_upper_bound = abs(param_lower_bound)

sigma_for_mutation = 0.0001
population_size = 500

non_linear_functions = [X_VARIABLE, X_VARIABLE**2, X_VARIABLE**3, 2**X_VARIABLE, 
                        sp.sin(X_VARIABLE), sp.cos(X_VARIABLE), sp.tan(X_VARIABLE), sp.tanh(X_VARIABLE), 
                        sp.Abs(X_VARIABLE)]

class GANONLinearOptimizer:
    min_delta = 0.001
    patience = 50

    def train(self, 
              x_train: DF, 
              y_train: Series, 
              x_val: DF | None = None, 
              y_val: Series | None = None, 
              epochs: int | None = None, 
              non_linearity: bool = False,
              crossover_method: str = 'none') -> None:  
        early_stop = False
        
        if epochs is None:
            if x_val is not None and y_val is not None:
                X_val = np.column_stack((np.ones(len(x_val)), x_val))     
                early_stop = True
            else:
                epochs = EPOCHS
        
        X = np.column_stack((np.ones(len(x_train)), x_train))
        
        self.min_train_mse = float('inf')
        self.min_val_mse = float('inf')
        
        number_of_features = X.shape[1]
        self.funcs = [X_VARIABLE for _ in range(number_of_features)]
        self.funcs = [sp.lambdify(X_VARIABLE, f, 'numpy') for f in self.funcs]
        
        if non_linearity:
            functions = [[np.random.choice(non_linear_functions) for _ in range(number_of_features)] for _ in range(population_size)]
 
        population = [np.random.uniform(param_lower_bound, param_upper_bound, number_of_features) for _ in range(population_size)]
        
        losses = [0 for _ in range(population_size)]
        
        self.epochs_performed = 0
        no_improvement = 0
        
        while True:
            lambdified_functions = [[sp.lambdify(X_VARIABLE, f, 'numpy') for f in funcs] for funcs in functions]
            self.epochs_performed += 1
            
            for i, solution in enumerate(population):
                if non_linearity:
                    y_pred_a = X*solution
                    
                    y_pred = np.sum(np.column_stack([f(y_pred_a[:, j]) for j, f in enumerate(lambdified_functions[i])]), axis=1)

                else:
                    y_pred = X @ solution
                
        
                losses[i] = mean_squared_error(y_pred, y_train)
 
            
            train_generation_min_mse = min(losses)
     
            
            self.min_train_mse = min(train_generation_min_mse, self.min_train_mse)
            
        
            if early_stop:
                for i, solution in enumerate(population):             
                    if non_linearity:
                        y_pred_a = X_val*solution
                        
                        y_pred = np.sum(np.column_stack([f(y_pred_a[:, j]) for j, f in enumerate(lambdified_functions[i])]), axis=1)

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

            top_50_percent_of_population = np.array([solution for _, solution in sorted(zip(losses, population))][:population_size//2])
            
            children = np.column_stack([GeneticAlgorithm.make_offspring(top_50_percent_of_population[:, j], crossover_method) 
                        for j in range(top_50_percent_of_population.shape[1])])
                        
            children = children.tolist()
            top_50_percent_of_population = top_50_percent_of_population.tolist()
            
            population = top_50_percent_of_population+children
            
            if non_linearity:
                top_50_percent_of_functions = np.array([solution for _, solution in sorted(zip(losses, functions))][:population_size//2])
                
                children = np.column_stack([GeneticAlgorithm.make_offspring(top_50_percent_of_functions[:, j], crossover_method) 
                        for j in range(top_50_percent_of_functions.shape[1])])
                
                children = children.tolist()
                top_50_percent_of_functions = top_50_percent_of_functions.tolist()
                
                functions = top_50_percent_of_functions + children
                
                    
        self.theta = population[0]
        
        if non_linearity:
            self.funcs = functions[0]
            self.funcs = [sp.lambdify(X_VARIABLE, f, 'numpy') for f in self.funcs]
    
    def predict(self, x: DF) -> NDArray:
        X = np.column_stack((np.ones(len(x)), x))
                        
        y_pred_a = X*self.theta
                        
        y_pred = np.sum(np.column_stack([f(y_pred_a[:, j]) for j, f in enumerate(self.funcs)]), axis=1)
        
        return y_pred
