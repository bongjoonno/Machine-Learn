from src.machine_learn.imports import np, sp
from src.machine_learn.constants import X_VARIABLE

crossover_methods = ['arithmetic', 'sbx', 'sbx_function']
selection_methods = ['threshold', 'tournament']

class GeneticAlgorithm:
    eta = 15
    
    min_of_feature = -1
    max_of_feature = 1
    
    distribution_len = 10
    uniform_feature_distribution = np.random.uniform(min_of_feature, max_of_feature, distribution_len)
    
    k_tournament_selections = 3
    
    
    @classmethod
    def get_crossover_dict(cls) -> dict[str, callable]:
        return {'arithmetic' : cls.arithmetic_crossover,
                                'sbx' : cls.sbx_crossover,
                                'sbx_function' : cls.sbx_function_crossover}
    
    @classmethod
    def get_selection_dict(cls) -> dict[str, callable]:
        return {'threshold' : cls.threshold_selection,
                'tournament' : cls.tournament_selection}
        
    def repopulate(self, solutions: list[float], fitness_scores: list[float], selection_method: str, crossover_method: str) -> list[float]:
        self.crossover_func = self.get_crossover_dict().get(crossover_method, None)
        
        if self.crossover_func is None:
            raise ValueError(f'crossover method must be one of the following: {crossover_methods}')
        
        self.selection_func = self.get_selection_dict().get(selection_method, None)
        
        if self.selection_func is None:
            raise ValueError(f'Selection method must be one of the following: {selection_methods}')

        selection = self.selection_func(solutions, fitness_scores)
        
        children = self.make_children(selection)
        
        return np.concatenate((selection, children))

    @staticmethod
    def threshold_selection(solutions: list[float], fitness_scores: list[float]) -> list[float]:
        return solutions[np.argsort(fitness_scores)]

    @staticmethod
    def tournament_selection(solutions: list[float], fitness_scores: list[float]):
        num_selections = len(solutions) // 2
        selected = []
        
        for _ in range(num_selections):
            rand_indices = np.random.randint(0, len(solutions), GeneticAlgorithm.k_tournament_selections)
            
            rand_solutions = [solutions[idx] for idx in rand_indices]
            rand_fitness_scores = [fitness_scores[idx] for idx in rand_indices]
            
            best_solution = rand_solutions[np.argmax(rand_fitness_scores)]
            
            selected.append(best_solution)
        
        return np.array(selected)

    def make_children(self, selection: list[float]):
        children = []
        
        for i in range(0, len(selection)-1, 2):
            parent_a = selection[i]
            parent_b = selection[i+1]

            child1, child2 = self.crossover_func(parent_a, parent_b)
            
            children.append(child1)
            children.append(child2)

        return np.array(children)
    
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