from src.machine_learn.imports import np, plt
from src.machine_learn.metrics import mean_squared_error
from src.machine_learn.genetic_algorithms import GeneticAlgorithm

# still need to determine how to determine bounds

param_lower_bound = -1_000_000_000
param_upper_bound = 1_000_000_000
population_size = 1000
generations = 1000

def optimize_parameter(x, y):
    x = x.to_numpy().flatten()
    y = y.to_numpy()
    
    n = len(x)
    
    '''
    1. generate random solutions
    2. select top 50%
    3. generate children from top 50%
    4. combine children and parent to make new population
    5. repeat 2-4 until convergence
    '''
    
    population = [np.random.uniform(param_lower_bound, param_upper_bound) for _ in range(population_size)]
    losses = [0 for _ in range(population_size)]
    
    # do weight first
    for _ in range(generations):
        for i, weight in enumerate(population):
            y_pred = x * weight
            
            mse = mean_squared_error(y_pred, y)
            losses[i] = mse


        top_50_percent_of_population = [solution for _, solution in sorted(zip(losses, population))][:n//2]
        
        children = GeneticAlgorithm.make_offspring(top_50_percent_of_population)
        
        population = top_50_percent_of_population + children
        
        print(mean_squared_error(x * top_50_percent_of_population[0], y))
    
    best_weight = population[0]
    print(population)
    best_preds = x * best_weight
    
    plt.plot(y_pred)
    plt.plot(y)
    plt.show()
    
    # do bias second
    population = [np.random.uniform(param_lower_bound, param_upper_bound) for _ in range(population_size)]
    
    for _ in range(generations):
        for i, bias in enumerate(population):
            y_pred = best_preds + bias
            
            mse = mean_squared_error(y_pred, y)
            losses[i] = mse


        top_50_percent_of_population = [solution for _, solution in sorted(zip(losses, population))][:n//2]
        
        children = GeneticAlgorithm.make_offspring(top_50_percent_of_population)
        
        population = top_50_percent_of_population + children
        
        print(mean_squared_error(x * top_50_percent_of_population[0], y))
    
    plt.plot(y_pred)
    plt.plot(y)
    plt.show()
    
    print(r2)