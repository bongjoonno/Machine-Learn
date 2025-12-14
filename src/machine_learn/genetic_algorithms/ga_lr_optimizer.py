from src.machine_learn.imports import np
from src.machine_learn.types import DF, Series
from src.machine_learn.models import LinearRegression
from src.machine_learn.metrics import mean_squared_error
from src.machine_learn.genetic_algorithms import GeneticAlgorithm

class GAlrOptimizer:
    learning_rate_low = 0.0001
    learning_rate_high = 0.1

    def __init__(self, model: LinearRegression, x_train: DF, y_train: Series, x_validation: DF, y_validation: Series) -> None:
        self.model = model
        self.x_train = x_train
        self.y_train = y_train
        self.x_validation = x_validation
        self.y_validation = y_validation
        
        self.population_size = 100
        self.generations = 100
        self.population = [0.0 for _ in range(self.population_size)]
        self.fitness_scores = [0.0 for _ in range(self.population_size)]         
    
    def optimize_lr(self) -> float:
        lowest_loss = float('inf')
        lowest_loss_lr = 0.0

        self.avg_fitness_scores_per_generation = []
        
        self.generate_initial_population()

        for _ in range(self.generations):
            self.fitness()

            generation_average_fitness_score = np.mean(self.fitness_scores)

            self.avg_fitness_scores_per_generation.append(generation_average_fitness_score)

            fitness_to_population: list[tuple[float, float]] = list(zip(self.fitness_scores, self.population))
           
            np.random.shuffle(fitness_to_population)
           
            fitness_to_population_sorted = sorted(fitness_to_population, key=lambda x: x[0])[:self.population_size//2]
            
            top_50_percent = [solution for _, solution in fitness_to_population_sorted]

            children = GeneticAlgorithm.make_offspring(top_50_percent)
            
            self.population = top_50_percent + children

            if fitness_to_population_sorted[0][0] < lowest_loss:
                lowest_loss = fitness_to_population_sorted[0][0]
                lowest_loss_lr = fitness_to_population_sorted[0][1]

        return lowest_loss_lr
            

    def generate_initial_population(self) -> None:
        for i in range(self.population_size):
            self.population[i] = np.random.uniform(GAlrOptimizer.learning_rate_low, GAlrOptimizer.learning_rate_high)
            
    def fitness(self) -> None:
        for i, learning_rate in enumerate(self.population):
            self.model.train(self.x_train, self.y_train, epochs = 2, learning_rate = learning_rate)
            y_pred = self.model.predict(self.x_validation)
            mse = mean_squared_error(y_pred, self.y_validation)
            self.fitness_scores[i] = mse