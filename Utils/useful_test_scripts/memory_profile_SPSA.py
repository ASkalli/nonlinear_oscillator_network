import numpy as np
import sys
sys.path.append('/home/anas/Desktop/Simulations/Training_NLO/nonlinear_oscillator_network/Utils')

import tracemalloc
import psutil, os   
import scipy
import time
from optimization_algorithms import *
import gc

import numpy as np, sys, psutil, os, gc, tracemalloc, time

def rss_mb():
    return psutil.Process(os.getpid()).memory_info().rss / 1e6

N_dim = 4045
dtype = np.float32  # change to np.float32 to halve data size
n_loops = 1

init_pos = np.random.randn(N_dim, 1).astype(dtype, copy=False)
spsa_optimizer = SPSA_opt(init_pos, alpha=3e-4, epsilon=1e-5)
adam_optimizer = AdamOptimizer(init_pos, lr=3e-4, beta1=0.9, beta2=0.9, epsilon=1e-8)

tracemalloc.start()
rss_before = rss_mb()

for _ in range(n_loops):
    

    # optimizers
    

    params_plus, params_minus = spsa_optimizer.perturb_parameters()
    reward_plus  = 0.4426910220384598
    reward_minus = 0.44264774668216705

    grad_spsa = spsa_optimizer.approximate_gradient(reward_plus, reward_minus)
    step = adam_optimizer.step(grad_spsa)
    current_params = spsa_optimizer.update_parameters_step(step)

    # _scalene_hold = [init_pos, params_plus, params_minus, grad_spsa, step, current_params]

    # force a GC cycle if we want a “post-op” snapshot
    gc.collect()

current, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()

rss_after = rss_mb()
rss_delta = rss_after - rss_before

# theoretical sizes (data buffers)
arrays = [("init_pos", init_pos),
          ("params_plus", params_plus),
          ("params_minus", params_minus),
          ("grad_spsa", grad_spsa),
          ("step", step),
          ("current_params", current_params)]

data_bytes = sum(a.nbytes for _, a in arrays)
py_overhead = sum(sys.getsizeof(a) for _, a in arrays)

print(f"dtype: {dtype}, itemsize: {np.dtype(dtype).itemsize} bytes")
for name, a in arrays:
    print(f"{name:16s}: shape={a.shape}, nbytes={a.nbytes/1e6:.3f} MB, py_obj_overhead≈{sys.getsizeof(a)} B")

print(f"\nTotal array data (explicit): {data_bytes/1e6:.3f} MB")
print(f"Total Python object overhead (explicit): ~{py_overhead/1e6:.3f} MB")

print(f"\ntracemalloc current Python-heap: {current/1e6:.3f} MB")
print(f"tracemalloc peak   Python-heap: {peak/1e6:.3f} MB")
print(f"Process RSS delta (real):       {rss_delta:.3f} MB")
print(f"Process RSS total (now):        {rss_after:.3f} MB")
