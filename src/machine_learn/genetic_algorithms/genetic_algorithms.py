from src.machine_learn.imports import np, sp
from src.machine_learn.constants import X_VARIABLE

crossover_methods = ['arithmetic', 'sbx', 'sbx_function']

class GeneticAlgorithm:
    eta = 15
    min_of_feature = -1
    max_of_feature = 1
    distribution_len = 10
    uniform_feature_distribution = np.random.uniform(min_of_feature, max_of_feature, distribution_len)
        
    @staticmethod
    def make_offspring(top_50_percent: list[float], crossover_method: str) -> list[float]:
        if crossover_method not in crossover_methods:
            raise ValueError(f'crossover method must be one of the following: {crossover_methods}')
        elif crossover_method == 'arithmetic':
            crossover_func = GeneticAlgorithm.arithmetic_crossover
        elif crossover_method == 'sbx':
            crossover_func = GeneticAlgorithm.sbx_crossover
        elif crossover_method == 'sbx_function':
            crossover_func = GeneticAlgorithm.sbx_function_crossover
            
        np.random.shuffle(top_50_percent)
        
        children = []

        for i in range(0, len(top_50_percent)-1, 2):
            parent_a = top_50_percent[i]
            parent_b = top_50_percent[i+1]

            child1, child2 = crossover_func(parent_a, parent_b)
            
            children.append(child1)
            children.append(child2)

        return children
    
    @staticmethod
    def arithmetic_crossover(parent_a: float, parent_b: float) -> tuple[float, float]:
        weight1 = np.random.uniform(0, 1)
        weight2 = 1 - weight1

        child_a_lr = (parent_a*weight1) + (parent_b*weight2)
        child_b_lr = (parent_a*weight2) + (parent_b*weight1)

        return child_a_lr, child_b_lr
    
    @staticmethod
    def sbx_crossover(parent_a: float, parent_b: float) -> tuple[float, float]:
        u = np.random.uniform(0, 1)
        
        x1 = min(parent_a, parent_b)
        x2 = max(parent_a, parent_b)

        exp = (1 / (GeneticAlgorithm.eta + 1))
        
        if u <= 0.5:
            beta = (2 * u) ** exp
        else:
            beta = (1 / (2 * (1 - u))) ** exp
        
        diff = x2 - x1
        mid = 0.5 * (x1 + x2)
        
        child_a =  mid - 0.5 * beta * diff
        child_b =  mid + 0.5 * beta * diff     

        return child_a, child_b
    
    @staticmethod
    def sbx_function_crossover(function_a: float, function_b: float) -> tuple[float, float]:
        u = np.random.uniform(0, 1)
        
        lambdified_function_a = sp.lambdify(X_VARIABLE, function_a, 'numpy')
        lambdified_function_b = sp.lambdify(X_VARIABLE, function_b, 'numpy')

        function_a_interval_mean = np.mean([lambdified_function_a(num) for num in GeneticAlgorithm.uniform_feature_distribution])
        function_b_interval_mean = np.mean([lambdified_function_b(num) for num in GeneticAlgorithm.uniform_feature_distribution])
        
        x1 = function_a if function_a_interval_mean <= function_b_interval_mean else function_b
        x2 = function_a if function_a_interval_mean >= function_b_interval_mean else function_b

        exp = (1 / (GeneticAlgorithm.eta + 1))
        
        if u <= 0.5:
            beta = (2 * u) ** exp
        else:
            beta = (1 / (2 * (1 - u))) ** exp
        
        diff = x2 - x1
        mid = 0.5 * (x1 + x2)
        
        child_a =  mid - 0.5 * beta * diff
        child_b =  mid + 0.5 * beta * diff     

        return child_a, child_b