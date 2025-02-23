import numpy as np

from pynput.keyboard import Key, Listener
from pynput import keyboard
import logging
import os
import threading

force_display = False


def run(
        individuals,
        simul,
        display,
        mutate,
        mutation_sigma,
        selectivity=3,
        n_generations=100,
        display_freq=10,
        verbose=False
):
    thread2 = threading.Thread(
        target=run_thread,
        args=(
            individuals,
            simul,
            display,
            mutate,
            mutation_sigma,
            selectivity,
            n_generations,
            display_freq,
            verbose
        )
    )
    thread2.start()

    with Listener(on_press=getKey) as listener:
        listener.join()


def getKey(key):
    global force_display
    if key == keyboard.Key.alt:
        print('Displaying the best of the current generation once finished...')
        force_display = True


def run_thread(
        individuals,
        simul,
        display,
        mutate,
        mutation_sigma,
        selectivity=3,
        n_generations=100,
        display_freq=10,
        verbose=False
):
    global force_display
    for gen in range(n_generations):
        if verbose:
            print('Generation', gen)
        results = np.zeros(len(individuals))
        for i, ind in enumerate(individuals):
            results[i] = simul(ind)
        rank = np.argsort(results)[::-1]
        if gen % display_freq == 0 or force_display:
            display(individuals[rank[0]])
            print(individuals[rank[0]])
            force_display = False
        children = []
        for i in rank[:len(individuals) // selectivity]:
            children += mutate(individuals[i], selectivity, sigma=mutation_sigma)
        individuals = children
