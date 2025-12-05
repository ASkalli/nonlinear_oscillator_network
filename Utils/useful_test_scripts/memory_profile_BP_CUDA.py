import os, psutil, gc, sys, time
import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
sys.path.append('/home/anas/Desktop/Simulations/Training_NLO/nonlinear_oscillator_network/Utils')

from optimization_algorithms import *
from NN_utils import *

# ----------------------------------------------------------
# Memory utilities
# ----------------------------------------------------------
def mem_mb():
    torch.cuda.synchronize()
    return torch.cuda.memory_allocated() / 1e6

def peak_mb():
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1e6

def reset_peak():
    # keep this consistent with what you do for PEPG/SPSA
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

def rss_mb():
    return psutil.Process(os.getpid()).memory_info().rss / 1e6

# ----------------------------------------------------------
# Env
# ----------------------------------------------------------
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
torch.set_num_threads(1)
assert torch.cuda.is_available(), "CUDA required"

# ----------------------------------------------------------
# Model + data
# ----------------------------------------------------------
RNN_params = {"N_in": 784, "N_out": 10, "N_neurons": 400, "N_layers": 3}
""""
model = Oscillator_RNN_parallel(params=RNN_params)
model.init_esn_weights(reservoir=True)
model.dt = 0.1
model.eps_int = 1e-4
model.alpha = 3
model.max_steps = 1
model.save_activations = False
"""
model = simple_FFNN(params=RNN_params)

dtype = torch.float32
device = torch.device("cuda")
model = model.to(device)

B = 1
x = torch.randn(B, RNN_params["N_in"], dtype=dtype, device=device)
y = torch.randint(0, RNN_params["N_out"], (B,), dtype=torch.long, device=device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=3e-4)

n_loops = 5
warmup_loops = 5

# ----------------------------------------------------------
# Warm-up: full training steps (not measured)
# ----------------------------------------------------------
for _ in range(warmup_loops):
    optimizer.zero_grad(set_to_none=True)
    out = model(x)
    loss = criterion(out, y)
    loss.backward()
    optimizer.step()

torch.cuda.synchronize()

# ----------------------------------------------------------
# Measurement: full backprop update, comparable to SPSA/PEPG
# ----------------------------------------------------------
if device.type == "cuda":
    reset_peak()
    mem_before = mem_mb()
else:
    mem_before = rss_mb()

for _ in range(n_loops):
    optimizer.zero_grad(set_to_none=True)
    out = model(x)
    loss = criterion(out, y)
    loss.backward()
    optimizer.step()

if device.type == "cuda":
    mem_after = mem_mb()
    peak = peak_mb()
else:
    mem_after = rss_mb()
    peak = mem_after

# ----------------------------------------------------------
# Report (comparable metrics)
# ----------------------------------------------------------
print(f"Device: {device}, dtype: {dtype}")
print(f"Memory before update: {mem_before:.3f} MB")
print(f"Memory after update:  {mem_after:.3f} MB")
print(f"Peak memory used:     {peak:.3f} MB")
print(f"Delta (after - before): {mem_after - mem_before:.3f} MB")
print(f"Update cost (peak - before): {peak - mem_before:.3f} MB")
