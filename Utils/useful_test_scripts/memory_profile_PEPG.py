import pickle
import numpy as np
import sys
sys.path.append('/home/anas/Desktop/Simulations/Training_NLO/nonlinear_oscillator_network/Utils')

import tracemalloc
import psutil, os   
import scipy
import time
from optimization_algorithms import *
import gc


#N_dim = int(avg_results[0]['n_params'])
N_dim = 10000   # adjust size as needed
n_loops = 1
# start memory tracing
tracemalloc.start()
for k in range(n_loops):
    pop_size = int((0.01*N_dim))
    pop_size = pop_size + 1 if pop_size % 2 else pop_size

    init_pos = np.random.randn(N_dim,1)
    rewards = np.random.randn(pop_size,1)

    PEPG_optimizer = PEPG_opt(N_dim, pop_size = pop_size, learning_rate=0.01, starting_mu=init_pos ,starting_sigma=1e-1)

    coordinates = PEPG_optimizer.ask()
    PEPG_optimizer.tell(rewards)

    _scalene_hold = [coordinates, rewards,PEPG_optimizer]
        # stop tracemalloc and print stats
    current, peak = tracemalloc.get_traced_memory()
    rss = psutil.Process(os.getpid()).memory_info().rss  # Resident set size (RAM held by process)
    gc.collect()
    print(k)
tracemalloc.stop()

print(f"Current allocated memory: {current / 1e6:.3f} MB")
print(f"Peak allocated memory (Python heap): {peak / 1e6:.3f} MB")
print(f"Total resident memory (process): {rss / 1e6:.3f} MB")

