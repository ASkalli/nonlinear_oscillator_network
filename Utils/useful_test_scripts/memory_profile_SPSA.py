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
N_dim = 4000   # adjust size as needed
n_loops = 10
# start memory tracing
tracemalloc.start()
for k in range(n_loops):
    # --- your workload ---
    init_pos = np.random.randn(N_dim, 1)

    spsa_optimizer = SPSA_opt(init_pos, alpha=3e-4, epsilon=1e-5)
    adam_optimizer = AdamOptimizer(init_pos, lr=3e-4, beta1=0.9, beta2=0.9, epsilon=1e-8)

    params_plus, params_minus = spsa_optimizer.perturb_parameters()

    reward_plus  = 0.4426910220384598
    reward_minus = 0.44264774668216705

    grad_spsa = spsa_optimizer.approximate_gradient(reward_plus, reward_minus)
    step = adam_optimizer.step(grad_spsa)
    current_params = spsa_optimizer.update_parameters_step(step)

    _scalene_hold = [init_pos, params_plus, params_minus, grad_spsa, step, current_params]

    # stop tracemalloc and print stats
    current, peak = tracemalloc.get_traced_memory()
    rss = psutil.Process(os.getpid()).memory_info().rss  # Resident set size (RAM held by process)
    gc.collect()
tracemalloc.stop()

print(f"Current allocated memory: {current / 1e6:.3f} MB")
print(f"Peak allocated memory (Python heap): {peak / 1e6:.3f} MB")
print(f"Total resident memory (process): {rss / 1e6:.3f} MB")
