from src.machine_learn.imports import np, random, sp
from src.machine_learn.constants import EPOCHS
from src.machine_learn.types import DF, Series, NDArray
from src.machine_learn.metrics import mean_squared_error, r_squared
from src.machine_learn.genetic_algorithms import GeneticAlgorithm

param_lower_bound = -0.8568
param_upper_bound = abs(param_lower_bound)

sigma_for_mutation = 0.0001
population_size = 4

x = sp.symbols('x')

non_linear_functions = [x, x**2, x**3, 2**x, 
                        sp.sin(x), sp.cos(x), sp.tan(x), sp.tanh(x), 
                        sp.Abs(x)]

class GANONLinearOptimizer:
    min_delta = 0.001
    patience = 0
    
    def train(self, 
              x_train: DF, 
              y_train: Series, 
              x_val: DF | None = None, 
              y_val: Series | None = None, 
              epochs: int | None = None, 
              mutate: bool = False, 
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

        if non_linearity:
            functions = [[np.random.choice(non_linear_functions) for _ in range(number_of_features)] for _ in range(population_size)]
 
        population = [np.random.uniform(param_lower_bound, param_upper_bound, number_of_features) for _ in range(population_size)]
        
        losses = [0 for _ in range(population_size)]
        
        self.epochs_performed = 0
        no_improvement = 0
        
        X = sp.Matrix(X)
        
        while True:
            self.epochs_performed += 1
            
            for i, solution in enumerate(population):
                solution = sp.Matrix(solution).T
                
                if non_linearity:
                    y_pred_a = sp.Matrix([
                        X[j, :].multiply_elementwise(solution)
                        for j in range(X.rows)
                        ])
                            
                    y_pred = sp.Matrix([
                        
                        sum([functions[i][j].subs(x, y_pred_a[k, j])
                            for j in range(y_pred_a.cols)
                        ])
                                       
                        for k in range(y_pred_a.rows)
                    ])

                else:
                    y_pred = X @ solution
                    
                losses[i] = mean_squared_error(y_pred, y_train)
            
            train_generation_min_mse = min(losses)
            
            self.min_train_mse = min(train_generation_min_mse, self.min_train_mse)
            
            if early_stop:
                for i, solution in enumerate(population):
                    solution = sp.Matrix(solution).T
                
                if non_linearity:
                    y_pred_a = sp.Matrix([
                        X_val[j, :].multiply_elementwise(solution)
                        for j in range(X_val.rows)
                        ])
                            
                    y_pred = sp.Matrix([
                        
                        sum([functions[i][j].subs(x, y_pred_a[k, j])
                            for j in range(y_pred_a.cols)
                        ])
                                       
                        for k in range(y_pred_a.rows)
                    ])

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
                
            top_50_percent_of_population = [solution for _, solution in sorted(zip(losses, population))][:population_size//2]
            
            children = []
            
            for i in range(number_of_features):
                params = []
                    
                for j in range(len(top_50_percent_of_population)):
                    params.append(top_50_percent_of_population[j][i])
                
                param_children = GeneticAlgorithm.make_offspring(params, crossover_method)
                
                if mutate:
                    for k in range(len(param_children)):
                        if np.random.random() < 0.01:
                            param_children[k] += random.gauss(0, sigma = sigma_for_mutation)
                
                children.append(param_children)
                        
                    
            
            children = np.array(children).T
            children = children.tolist()
            
            population = top_50_percent_of_population+children
            
            if non_linearity:
                top_50_percent_of_functions = [solution for _, solution in sorted(zip(losses, functions))][:population_size//2]
                functions = top_50_percent_of_functions * 2
                    
        self.theta = population[0]
        self.funcs = functions[0]
    
    def predict(self, x: DF) -> NDArray:
        X = np.column_stack((np.ones(len(x)), x))
        
        y = X * self.theta
        y = np.sum(np.column_stack([f(y[:, i]) for i, f in enumerate(self.funcs)]), axis=1)
        return y