from src.machine_learn.imports import np
from src.machine_learn.models.linear_regression import LinearRegression

class GAHParamOptimizer:
    epoch_low = 1
    epoch_high = 5_000

    learning_rate_low = 0.00001
    learning_rate_high = 1

    def __init__(self, population_size = 8):
        self.population_size = population_size
        self.population = [0 for _ in range(self.population_size)]
        self.fitness_scores = [0 for _ in range(self.population_size)]

    def optimize(self, linear_regression_model: LinearRegression, x_validation, y_validation, generations = 100):
        self.generate_population()
        self.model = linear_regression_model
        self.x_validation = x_validation
        self.y_validation = y_validation

        self.fitness()
        population_sorted_by_fitness = [chromosome for _, chromosome in sorted(zip(self.fitness_scores, self.population))]
        self.top_50_percent = population_sorted_by_fitness[:self.population_size//2]
        print(population_sorted_by_fitness)
        print(self.top_50_percent)
        self.repopulate()
    
    def repopulate(self):
        np.random.shuffle(self.top_50_percent)
        
        children = []

        for i in range(0, len(self.top_50_percent)-1, 2):
            parent1 = self.top_50_percent[i]
            parent2 = self.top_50_percent[i+1]
            children.append(self.crossover(parent1, parent2))
    
    def crossover(self, parent_a, parent_b):
        weight1 = np.random.random()
        weight2 = 1 - weight1


        child_a_epochs = (parent_a[0]*weight1) + (parent_b[0]*weight2)
        child_b_epochs = (parent_a[0]*weight2) + (parent_b[0]*weight1)

        child_a_lr = (parent_a[1]*weight1) + (parent_b[1]*weight2)
        child_b_lr = (parent_a[1]*weight2) + (parent_b[1]*weight1)

        return ((child_a_epochs, child_a_lr), (child_b_epochs, child_b_lr))

    def fitness(self):
        for i, (epochs, learning_rate) in enumerate(self.population):
            self.model.train(self.x_validation, self.y_validation, epochs, learning_rate)
            self.fitness_scores[i] = self.model.min_loss

    def generate_population(self):
        for i in range(self.population_size):
            random_epochs = np.random.randint(GAHParamOptimizer.epoch_low, GAHParamOptimizer.epoch_high)
            random_learning_rate = np.random.uniform(GAHParamOptimizer.learning_rate_low, GAHParamOptimizer.learning_rate_high)
            self.population[i] = ((random_epochs, random_learning_rate))







# generate initial population of random (epoch, learning_rate) tuple pairs
# random epochs in range (1, 100_000) and learning_rate from (0.000001, 0.5)
# evaluate each by training and getting fitness function