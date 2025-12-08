from src.machine_learn.imports import np, plt
from src.machine_learn.metrics import mean_squared_error, r_squared
from src.machine_learn.genetic_algorithms import GeneticAlgorithm

# still need to determine how to determine bounds

population_size = 200
generations = 100

param_lower_bound = -1
param_upper_bound = abs(param_lower_bound)

param_bound_increment = ((2*param_upper_bound)+1)/population_size

def optimize_weight(x, y):
    population = [np.random.uniform(param_lower_bound, param_upper_bound) for _ in range(population_size)]
    losses = [0 for _ in range(population_size)]
    
    for _ in range(generations):
        for i, weight in enumerate(population):
            y_pred = x * weight
            
            mse = mean_squared_error(y_pred, y)
            losses[i] = mse


        top_50_percent_of_population = [solution for _, solution in sorted(zip(losses, population))][:population_size//2]
        
        children = GeneticAlgorithm.make_offspring(top_50_percent_of_population)
        
        population = top_50_percent_of_population + children
    
    best_weight = population[0]
    return best_weight


def optimize_bias(x, y):
    population = [np.random.uniform(param_lower_bound, param_upper_bound) for _ in range(population_size)]
    losses = [0 for _ in range(population_size)]
    
    for _ in range(generations):
        for i, bias in enumerate(population):
            y_pred = x + bias
            
            mse = mean_squared_error(y_pred, y)
            losses[i] = mse


        top_50_percent_of_population = [solution for _, solution in sorted(zip(losses, population))][:population_size//2]
        
        children = GeneticAlgorithm.make_offspring(top_50_percent_of_population)
        
        population = top_50_percent_of_population + children
    
    best_bias = population[0]
    
    return best_bias

def ga_optimize_params(x, y):
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
            
            children.append(GeneticAlgorithm.make_offspring(params))
        
        children = np.array(children)
        children = children.T
        children = children.tolist()
        
        population = top_50_percent_of_population + children
    
    return population[0]