import os, psutil, gc, sys, time
import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import sys
sys.path.append('/home/anas/Desktop/Simulations/Training_NLO/nonlinear_oscillator_network/Utils')

import tracemalloc
import psutil, os
import gc
from optimization_algorithms import *
from NN_utils import *


# ----------------------------------------------------------
# Memory utilities
# ----------------------------------------------------------
def rss_mb():
    """Return process memory in MB."""
    return psutil.Process(os.getpid()).memory_info().rss / 1e6

# Limit threading noise
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
torch.set_num_threads(1)

# ----------------------------------------------------------
# Model setup (our ODE RNN class)
# ----------------------------------------------------------

n_neurons = 250  # sweep this to study scaling

RNN_params = {
    "N_in": 784,
    "N_out": 10,
    "N_neurons": n_neurons,
    "N_layers": 3
}

model = Oscillator_RNN_parallel(params=RNN_params)

# Any special init you already use
model.init_esn_weights(reservoir=True)
model.dt = 0.1
model.eps_int = 1e-4
model.alpha = 3
model.max_steps = 40
model.save_activations = False

# ----------------------------------------------------------
# Dummy data + optimizer + loss
# ----------------------------------------------------------
dtype = torch.float32
device = torch.device("cpu")
B = 64  # batch size

x = torch.randn(B, RNN_params["N_in"], dtype=dtype, device=device)
y = torch.randint(0, RNN_params["N_out"], (B,), dtype=torch.long, device=device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=3e-4)

# Count parameters
n_params = model.count_parameters()
itemsize = torch.finfo(dtype).bits // 8
print(f"Model parameters: {n_params:,} | dtype: {dtype}, itemsize={itemsize} bytes")

# ----------------------------------------------------------
# Measurement: one training update
# ----------------------------------------------------------
# Warmup to allocate optimizer state
optimizer.zero_grad(set_to_none=True)
_ = model(x)
_ = criterion(_, y)
_.backward()
optimizer.step()
optimizer.zero_grad(set_to_none=True)
gc.collect()

# Now measure one update cleanly
rss_before = rss_mb()

optimizer.zero_grad(set_to_none=True)
out = model(x)
loss = criterion(out, y)
loss.backward()
optimizer.step()
optimizer.zero_grad(set_to_none=True)

del out, loss
gc.collect()

rss_after = rss_mb()
rss_delta = rss_after - rss_before

# ----------------------------------------------------------
# Report
# ----------------------------------------------------------
print(f"\nProcess RSS delta (real): {rss_delta:.3f} MB")
print(f"Process RSS total (now):  {rss_after:.3f} MB")

# Optional theoretical baseline
print(f"Theoretical param data: {(n_params * itemsize) / 1e6:.3f} MB")
