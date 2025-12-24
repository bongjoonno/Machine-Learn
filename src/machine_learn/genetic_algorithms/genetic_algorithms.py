from src.machine_learn.imports import np, cp

class GeneticAlgorithm:
    eta = 15
    
    @staticmethod
    def make_offspring(top_50_percent: list[float]) -> list[float]:
        np.random.shuffle(top_50_percent)
        
        children = []

        for i in range(0, len(top_50_percent)-1, 2):
            parent_a = top_50_percent[i]
            parent_b = top_50_percent[i+1]

            child1, child2 = GeneticAlgorithm.arithmetic_crossover(parent_a, parent_b)
            
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

        exp = (1 / (GeneticAlgorithm.eta + 1))
        if u <= 0.5:
            beta = (2*u)**exp
        else:
            beta = (1 / (2(1-u)))^exp
            
        child_a =  + (parent_b*weight2)
        child_b = (parent_a*weight2) + (parent_b*weight1)

        return child_a_lr, child_b_lr