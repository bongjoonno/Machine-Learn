from src.machine_learn.imports import np

def arithmetic_crossover(parent_a: float, parent_b: float) -> tuple[float, float]:
    weight1 = np.random.uniform(0, 1)
    weight2 = 1 - weight1

    child_a_lr = (parent_a*weight1) + (parent_b*weight2)
    child_b_lr = (parent_a*weight2) + (parent_b*weight1)

    return child_a_lr, child_b_lr