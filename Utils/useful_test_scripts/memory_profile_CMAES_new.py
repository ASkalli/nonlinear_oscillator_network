#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Measure memory cost of a CMA-ES update on CPU using:
  - Peak RSS over baseline (robust for NumPy C-allocations)
  - tracemalloc (Python-heap diagnostics; may under/overcount vs native)
"""

import os, sys, gc, platform, ctypes
import numpy as np
import psutil, tracemalloc

# ---  local utils path (adjust if needed) ---
sys.path.append('/home/anas/Desktop/Simulations/Training_NLO/nonlinear_oscillator_network/Utils')
from optimization_algorithms import *   # expects CMA_opt

# ------------------------------------------------
# Utilities
# ------------------------------------------------
def rss_mb() -> float:
    """Return process RSS (resident set size) in MB."""
    return psutil.Process(os.getpid()).memory_info().rss / 1e6

def malloc_trim():
    """Ask glibc to return free heap pages to OS (Linux only)."""
    if platform.system() == "Linux":
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass

def report_array(name, arr):
    """Print array metadata safely."""
    if isinstance(arr, np.ndarray):
        print(f"{name:16s}: shape={arr.shape}, dtype={arr.dtype}, "
              f"nbytes={arr.nbytes/1e6:.3f} MB, py_obj_overhead≈{sys.getsizeof(arr)} B")

# ------------------------------------------------
# Parameters
# ------------------------------------------------
N_dim   = 25720          # search dimension
dtype   = np.float64     # prefer float32 to halve memory vs float64
n_loops = 1              # number of ask/tell updates to measure

# Population size (even)
pop_size = max(int(0.01 * N_dim), 10)
pop_size = pop_size + 1 if (pop_size % 2) else pop_size

print(f"dtype: {dtype}, itemsize: {np.dtype(dtype).itemsize} bytes")
print(f"N_dim={N_dim}, pop_size={pop_size}, n_loops={n_loops}")

# ------------------------------------------------
# Initialization (outside measurement)
# ------------------------------------------------
rng = np.random.default_rng(123)

# Initial mean (column vector). Keep dtype consistent.
init_pos = rng.standard_normal((N_dim, 1), dtype=dtype)

# Rewards for tell(); shape must match CMA implementation.
rewards = rng.standard_normal((pop_size, 1), dtype=dtype)

# Create optimizer. NOTE: If CMA_opt internally uses float64, memory doubles.
CMA_optimizer = CMA_opt(
    N_dim,
    pop_size,
    select_pop=int(pop_size/2),
    sigma_init=0.01,
    mean_init=init_pos
)

# ------------------------------------------------
# Warm-up: allocate internal state once
# ------------------------------------------------
_ = CMA_optimizer.ask()
CMA_optimizer.tell(rewards)
del _
gc.collect(); malloc_trim()

# ------------------------------------------------
# Measurement: ask -> tell (peak RSS during block)
# ------------------------------------------------
tracemalloc.start()
gc.collect(); malloc_trim()

rss_before = rss_mb()
peak_rss = rss_before

for _ in range(n_loops):
    coordinates = CMA_optimizer.ask()    # expected (pop_size x N_dim) or similar
    CMA_optimizer.tell(rewards)

    # Track instantaneous peak RSS
    r = rss_mb()
    if r > peak_rss:
        peak_rss = r

# Optional cleanup before final snapshots
del coordinates
gc.collect(); malloc_trim()

current_py, peak_py = tracemalloc.get_traced_memory()
tracemalloc.stop()

rss_after = rss_mb()

# ------------------------------------------------
# Reporting
# ------------------------------------------------
print("\n--- Visible arrays ---")
report_array("init_pos", init_pos)
try:
    report_array("coordinates", coordinates)   # may be deleted above; guarded
except NameError:
    pass
report_array("rewards", rewards)

# Try to introspect common internal arrays (if CMA_opt exposes them)
for attr in ["mu", "sigma", "population", "eps", "noise", "cov", "C", "D", "B"]:
    val = getattr(CMA_optimizer, attr, None)
    if isinstance(val, np.ndarray):
        report_array(attr, val)

print("\n--- Memory results ---")
print(f"Process RSS before:             {rss_before:.3f} MB")
print(f"Process RSS after:              {rss_after:.3f} MB")
print(f"Process RSS PEAK (during loop): {peak_rss:.3f} MB")
print(f"Δ Peak RSS over baseline:       {peak_rss - rss_before:.3f} MB   <-- use this")
print(f"tracemalloc current (Python):   {current_py/1e6:.3f} MB")
print(f"tracemalloc peak    (Python):   {peak_py/1e6:.3f} MB")

print("\nNotes:")
print("- Δ Peak RSS over baseline is robust and won’t be negative; it reflects true max footprint of the update(s).")
print("- If any arrays show dtype=float64,then it means our CMA pipeline is upcasting; expect ~2× memory vs float32.")
print("- tracemalloc measures Python allocations, not NumPy’s C buffers; treat it as diagnostic, not ground truth.")
print("- malloc_trim() helps reduce allocator noise on Linux; harmless elsewhere.")
