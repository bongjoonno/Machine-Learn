from src.machine_learn.imports import np
from src.machine_learn.metrics import mean_squared_error
# still need to determine how to determine bounds

param_lower_bound = -1000
param_upper_bound = 1000
population_size = 100

def optimize_parameter(x, y):
    x = x.to_numpy()
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

    
    # measure loss
    losses = []
    
    for solution in population:
        y_pred = x * solution
        mse = mean_squared_error(y_pred, y)
        losses.append(mse)


    top_50_percent_of_population = [solution for _, solution in sorted(zip(losses, population))][:n//2]