from math import log
from scipy.stats import entropy
from collections import Counter

def get_entropy(items):
    n = len(items)

    items_count = Counter(items)

    entropy = 0

    for item, count in items_count.items():
        prob = count / n
        entropy += prob * log(1 / prob)

    return entropy