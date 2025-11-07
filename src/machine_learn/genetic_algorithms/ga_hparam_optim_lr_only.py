from src.machine_learn.imports import np, tqdm
from src.machine_learn.models.linear_regression import LinearRegression

class GAHParamOptim:
    learning_rate_low = 0.0001
    learning_rate_high = 0.1
    optimization_runs = 10

    def __init__(self, population_size = 8):
        self.population_size = population_size
        self.population = [0 for _ in range(self.population_size)]
        self.fitness_scores = [0 for _ in range(self.population_size)]

    def optimize(self, model: LinearRegression, x_validation, y_validation, generations = 10):
        solution_to_loss = {}
        self.model = model
        self.x_validation = x_validation
        self.y_validation = y_validation
        self.avg_fitness_scores_per_generation = [0 for _ in range(generations)]

        for _ in range(GAHParamOptim.optimization_runs):
            self.generate_population()

            for i in range(generations):
                self.fitness()

                generation_average_fitness_score = np.mean(self.fitness_scores)
    
                self.avg_fitness_scores_per_generation[i] = generation_average_fitness_score

                # biasin toward smaller learning rates, needs fix
                fitness_to_population = list(zip(self.fitness_scores, self.population))
                np.random.shuffle(fitness_to_population)
                fitness_to_population_sorted = sorted(fitness_to_population, key=lambda x: x[0])[:self.population_size//2]
                top_50_percent = [solution for _, solution in fitness_to_population_sorted]

                children = GAHParamOptim.make_offspring(top_50_percent)
                self.population = top_50_percent + children

                for item in fitness_to_population_sorted: print(item)
                print('\n')

                
        
        return min(solution_to_loss, key=solution_to_loss.get)
            



    
    @staticmethod
    def make_offspring(top_50_percent):
        np.random.shuffle(top_50_percent)
        
        children = []

        for i in range(0, len(top_50_percent)-1, 2):
            parent_a = top_50_percent[i]
            parent_b = top_50_percent[i+1]

            child1, child2 = GAHParamOptim.crossover(parent_a, parent_b)
            
            children.append(child1)
            children.append(child2)

        return children
    
    @staticmethod
    def crossover(parent_a, parent_b):
        lr_weight1 = np.random.uniform(0, 1)
        lr_weight2 = 1 - lr_weight1

        child_a_lr = (parent_a*lr_weight1) + (parent_b*lr_weight2)
        child_b_lr = (parent_a*lr_weight2) + (parent_b*lr_weight1)

        return child_a_lr, child_b_lr

    def fitness(self):
        for i, learning_rate in enumerate(self.population):
            self.model.train(self.x_validation, self.y_validation, epochs = 50, learning_rate = learning_rate)
            self.fitness_scores[i] = self.model.min_loss

    def generate_population(self):
        for i in range(self.population_size):
            random_learning_rate = np.random.uniform(GAHParamOptim.learning_rate_low, GAHParamOptim.learning_rate_high)
            self.population[i] = random_learning_rate







# generate initial population of random (epoch, learning_rate) tuple pairs
# random epochs in range (1, 100_000) and learning_rate from (0.000001, 0.5)
# evaluate each by training and getting fitness function