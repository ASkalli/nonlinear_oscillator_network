import numpy as np
import sys
sys.path.append('/home/anas/Desktop/Simulations/Training_NLO/nonlinear_oscillator_network/Utils')

import tracemalloc
import psutil, os
import gc
from optimization_algorithms import *

def rss_mb():
    return psutil.Process(os.getpid()).memory_info().rss / 1e6

# -------------------------------
# Params
# -------------------------------
N_dim = 8180             # adjust as needed
dtype = np.float32         # switch to np.float64 to compare but doesn't work for now ... just divide by 2
n_loops = 1                # number of ask/tell updates to run (and measure)

# -------------------------------
# Initialization (OUTSIDE measurement)
# -------------------------------
init_pos = np.random.randn(N_dim, 1).astype(dtype, copy=False)

# choose an even pop_size (PEPG typically uses mirrored sampling)
pop_size = np.max([int(0.01 * N_dim),10])
pop_size = pop_size + 1 if (pop_size % 2) else pop_size

# dummy rewards for the tell() step (shape must match CMA implementation)
rewards = np.random.randn(pop_size, 1).astype(dtype, copy=False)

CMA_optimizer = CMA_opt(N_dim, pop_size, select_pop=int(pop_size/2), sigma_init=0.01, mean_init=init_pos)

# -------------------------------
# Measurement (ask -> tell)
# -------------------------------
tracemalloc.start()
rss_before = rss_mb()

for _ in range(n_loops):
    coordinates = CMA_optimizer.ask()     #  returns population samples (pop_size x N_dim)
    #give dummy rewards
    CMA_optimizer.tell(rewards)

    
    # del coordinates
    gc.collect()

current, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()

rss_after = rss_mb()
rss_delta = rss_after - rss_before

# -------------------------------
# Reporting (arrays we can see)
# -------------------------------
def report_array(name, arr):
    if isinstance(arr, np.ndarray):
        print(f"{name:16s}: shape={arr.shape}, dtype={arr.dtype}, nbytes={arr.nbytes/1e6:.3f} MB, py_obj_overhead≈{sys.getsizeof(arr)} B")

print(f"dtype: {dtype}, itemsize: {np.dtype(dtype).itemsize} bytes")

# Known arrays
report_array("init_pos", init_pos)
try:
    report_array("coordinates", coordinates)
except NameError:
    pass
report_array("rewards", rewards)

# Try to report common internal state if exposed by class (optional, safe checks)
for attr in ["mu", "sigma", "population", "eps", "noise"]:
    val = getattr(CMA_optimizer, attr, None)
    if isinstance(val, np.ndarray):
        report_array(attr, val)

print(f"\ntracemalloc current Python-heap: {current/1e6:.3f} MB")
print(f"tracemalloc peak   Python-heap: {peak/1e6:.3f} MB")
print(f"Process RSS delta (real):       {rss_delta:.3f} MB")
print(f"Process RSS total (now):        {rss_after:.3f} MB")
