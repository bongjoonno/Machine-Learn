from src.machine_learn.imports import np, plt, random
from src.machine_learn.metrics import mean_squared_error, r_squared
from src.machine_learn.genetic_algorithms import GeneticAlgorithm

# still need to determine how to determine bounds

param_lower_bound = -0.8568
sigma_for_mutation = 0.0001

def ga_optimize_params(x, y, population_size, mutate: bool = True):
    param_upper_bound = abs(param_lower_bound)
    number_of_features = x.shape[1]
    population = [np.random.uniform(param_lower_bound, param_upper_bound, number_of_features) for _ in range(population_size)]
        
    losses = [0 for _ in range(population_size)]
    
    min_mse = float('inf')
    
    while True:
        for i, solution in enumerate(population):
            y_pred = x @ solution
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
    
    return population[0]