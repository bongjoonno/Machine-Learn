from src.machine_learn.imports import np, tqdm
from src.machine_learn.models.linear_regression import LinearRegression
class GAHParamOptimizer:
    epoch_low = 1
    epoch_high = 5_000

    learning_rate_low = 0.000001
    learning_rate_high = 1

    def __init__(self, population_size = 8):
        self.population_size = population_size
        self.population = [0 for _ in range(self.population_size)]
        self.fitness_scores = [0 for _ in range(self.population_size)]

    def optimize(self, linear_regression_model: LinearRegression, x_validation, y_validation, generations = 5):
        self.generate_population()
        self.model = linear_regression_model
        self.x_validation = x_validation
        self.y_validation = y_validation
        self.avg_fitness_scores_per_generation = [0 for _ in range(generations)]
        self.lowest_loss = float('inf')

        for i in tqdm(range(generations)):
            self.fitness()

            generation_average_fitness_score = np.mean(self.fitness_scores)
   
            self.avg_fitness_scores_per_generation[i] = generation_average_fitness_score
            
            population_sorted_by_fitness = [chromosome for _, chromosome in sorted(zip(self.fitness_scores, self.population))]
            top_50_percent = population_sorted_by_fitness[:self.population_size//2]

            children = GAHParamOptimizer.make_offspring(top_50_percent)
            self.population = top_50_percent + children

            if self.fitness_scores[0] < self.lowest_loss:
                self.lowest_loss_solution = population_sorted_by_fitness[0]
            
            print(self.lowest_loss, self.lowest_loss_solution)


    
    @staticmethod
    def make_offspring(top_50_percent):
        np.random.shuffle(top_50_percent)
        
        children = []

        for i in range(0, len(top_50_percent)-1, 2):
            parent_a = top_50_percent[i]
            parent_b = top_50_percent[i+1]

            child1, child2 = GAHParamOptimizer.crossover(parent_a, parent_b)
            
            children.append(child1)
            children.append(child2)

        return children
    
    @staticmethod
    def crossover(parent_a, parent_b):
        epochs_weight1 = np.random.uniform(0, 1)
        epochs_weight2 = 1 - epochs_weight1

        lr_weight1 = np.random.uniform(0, 1)
        lr_weight2 = 1 - lr_weight1


        child_a_epochs = int((parent_a[0]*epochs_weight1) + (parent_b[0]*epochs_weight2))
        child_b_epochs = int((parent_a[0]*epochs_weight2) + (parent_b[0]*epochs_weight1))

        child_a_lr = (parent_a[1]*lr_weight1) + (parent_b[1]*lr_weight2)
        child_b_lr = (parent_a[1]*lr_weight2) + (parent_b[1]*lr_weight1)

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