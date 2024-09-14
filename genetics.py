import numpy as np


def run(individuals, simul, display, mutate, selectivity=3, n_generations=100, display_freq=10):
    for gen in range(n_generations):
        results = np.zeros(len(individuals))
        for ind in individuals:
            results[ind] = simul(ind)
        rank = np.argsort(results)[::-1]
        if gen % display_freq == 0:
            display(individuals[rank[0]])
        children = []
        for i in rank[:len(individuals) // selectivity]:
            children += mutate(individuals[i], selectivity)
        individuals = children
