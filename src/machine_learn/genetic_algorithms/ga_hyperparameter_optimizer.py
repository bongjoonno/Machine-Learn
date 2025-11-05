

class GeneticHParamOptimizer:
    epoch_low = 1
    epoch_high = 100_000

    learning_rate_low = 0.000001
    learning_rate_high = 0.5

    def generate_population(size = 100):
        population = []

        for _ in range(size):
            pass
        
        pass







# generate initial population of random (epoch, learning_rate) tuple pairs
# random epochs in range (1, 100_000) and learning_rate from (0.000001, 0.5)
# evaluate each by training and getting fitness function