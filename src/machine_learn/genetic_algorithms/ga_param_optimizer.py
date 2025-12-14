from src.machine_learn.imports import np, random
from src.machine_learn.types import DF, Series, NDArray
from src.machine_learn.metrics import mean_squared_error, r_squared
from src.machine_learn.genetic_algorithms import GeneticAlgorithm

# still need to determine how to determine bounds

param_lower_bound = -0.8568
param_upper_bound = abs(param_lower_bound)

sigma_for_mutation = 0.0001
population_size = 1000

class GAOptimizer:
    def train(self, x_train: DF, y: Series, mutate: bool = True):
        X = np.column_stack((np.ones(len(x_train)), x_train))
        number_of_features = X.shape[1]

        population = [np.random.uniform(param_lower_bound, param_upper_bound, number_of_features) for _ in range(population_size)]
            
        losses = [0 for _ in range(population_size)]
        
        min_mse = float('inf')
        
        while True:
            for i, solution in enumerate(population):
                y_pred = X @ solution
                losses[i] = mean_squared_error(y_pred, y)
            
            generation_min_mse = min(losses)
            
            if generation_min_mse >= min_mse:
                break
            else:
                min_mse = generation_min_mse
                
            top_50_percent_of_population = [solution for _, solution in sorted(zip(losses, population))][:population_size//2]
            
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
        
        self.theta = population[0]
    
    def predict(self, x: DF) -> NDArray:
        X = np.column_stack((np.ones(len(x)), x))
        y = X @ self.theta
        return y