from src.machine_learn.imports import np
from src.machine_learn.models.linear_regression import LinearRegression

class GAHParamOptimizer:
    epoch_low = 1
    epoch_high = 1000

    learning_rate_low = 0.001
    learning_rate_high = 0.5

    def __init__(self, population_size = 100):
        self.population_size = population_size
        self.population = [0 for _ in range(self.population_size)]
        self.fitness_scores = [0 for _ in range(self.population_size)]

    def optimize(self, linear_regression_model: LinearRegression, x_validation, y_validation, generations = 100):
        self.generate_population()
        self.model = linear_regression_model
        self.x_validation = x_validation
        self.y_validation = y_validation

        for _ in range(generations):
            self.fitness()
    
    def generate_population(self):
        for i in range(self.population_size):
            random_epochs = np.random.randint(GAHParamOptimizer.epoch_low, GAHParamOptimizer.epoch_high)
            random_learning_rate = np.random.uniform(GAHParamOptimizer.learning_rate_low, GAHParamOptimizer.learning_rate_high)
            self.population[i] = ((random_epochs, random_learning_rate))


    def fitness(self):
        for i, epochs, learning_rate in enumerate(self.population):
            self.model.train(self.x_validation, self.y_validation, epochs, learning_rate)
            self.fitness_scores[i] = self.model.min_loss







# generate initial population of random (epoch, learning_rate) tuple pairs
# random epochs in range (1, 100_000) and learning_rate from (0.000001, 0.5)
# evaluate each by training and getting fitness function