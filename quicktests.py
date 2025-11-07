import numpy as np

def crossover(parent_a, parent_b):
        epochs_weight1 = np.random.random()
        epochs_weight2 = 1 - epochs_weight1

        lr_weight1 = np.random.random()
        lr_weight2 = 1 - lr_weight1


        child_a_epochs = int((parent_a[0]*epochs_weight1) + (parent_b[0]*epochs_weight2))
        child_b_epochs = int((parent_a[0]*epochs_weight2) + (parent_b[0]*epochs_weight1))

        child_a_lr = (parent_a[1]*lr_weight1) + (parent_b[1]*lr_weight2)
        child_b_lr = (parent_a[1]*lr_weight2) + (parent_b[1]*lr_weight1)

        return ((child_a_epochs, child_a_lr), (child_b_epochs, child_b_lr))


print(crossover((500, 0.5), (1000, 1)))
