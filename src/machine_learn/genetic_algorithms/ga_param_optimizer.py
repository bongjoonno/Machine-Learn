from src.machine_learn.imports import np, plt
from src.machine_learn.metrics import mean_squared_error, r_squared
from src.machine_learn.genetic_algorithms import GeneticAlgorithm

# still need to determine how to determine bounds

param_lower_bound = -1_000_000_000
param_upper_bound = 1_000_000_000
population_size = 100
generations = 100

def optimize_weight_and_bias_seperately(x, y):
    n = len(x)
    
    population = [np.random.uniform(param_lower_bound, param_upper_bound) for _ in range(population_size)]
    losses = [0 for _ in range(population_size)]
    
    # optimize weight first
    for _ in range(generations):
        for i, weight in enumerate(population):
            y_pred = x * weight
            
            mse = mean_squared_error(y_pred, y)
            losses[i] = mse


        top_50_percent_of_population = [solution for _, solution in sorted(zip(losses, population))][:n//2]
        
        children = GeneticAlgorithm.make_offspring(top_50_percent_of_population)
        
        population = top_50_percent_of_population + children
    
    best_weight = population[0]
    best_preds = x * best_weight

    
    # optimize bias second
    population = [np.random.uniform(param_lower_bound, param_upper_bound) for _ in range(population_size)]
    
    for _ in range(generations):
        for i, bias in enumerate(population):
            y_pred = best_preds + bias
            
            mse = mean_squared_error(y_pred, y)
            losses[i] = mse


        top_50_percent_of_population = [solution for _, solution in sorted(zip(losses, population))][:n//2]
        
        children = GeneticAlgorithm.make_offspring(top_50_percent_of_population)
        
        population = top_50_percent_of_population + children
    
    best_bias = population[0]
    
    return (best_weight, best_bias)
    
    