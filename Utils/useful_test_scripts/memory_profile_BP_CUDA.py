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

# model init params
model.init_esn_weights(reservoir=True)
model.dt = 0.1
model.eps_int = 1e-4
model.alpha = 3
model.max_steps = 1
model.save_activations = False

# ----------------------------------------------------------
# Dummy data + optimizer + loss
# ----------------------------------------------------------
dtype = torch.float32
device = torch.device("cpu")
B = 1000  # batch size

x = torch.randn(B, RNN_params["N_in"], dtype=dtype, device=device)
y = torch.randint(0, RNN_params["N_out"], (B,), dtype=torch.long, device=device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=3e-4)

# Count parameters
n_params = model.count_parameters()
itemsize = torch.finfo(dtype).bits // 8
print(f"Model parameters: {n_params:,} | dtype: {dtype}, itemsize={itemsize} bytes")

# ----------------------------------------------------------
# Move model & data to CUDA
# ----------------------------------------------------------
# ----------------------------------------------------------
# CUDA memory breakdown: activations, backward, optimizer
# ----------------------------------------------------------
assert torch.cuda.is_available(), "CUDA is required for this measurement."
device = torch.device("cuda")
model = model.to(device)
x = x.to(device, non_blocking=True)
y = y.to(device, non_blocking=True)

def mem_mb():
    return torch.cuda.memory_allocated() / 1e6
def peak_mb():
    return torch.cuda.max_memory_allocated() / 1e6

# ---------- Warm-up ----------
optimizer.zero_grad(set_to_none=True)
(loss := criterion(model(x), y)).backward()
optimizer.step()
optimizer.zero_grad(set_to_none=True)
torch.cuda.empty_cache()
torch.cuda.synchronize()

print("\n=== Full training memory breakdown (CUDA) ===")

# ---------- (A) Inference baseline (no grad) ----------
torch.cuda.empty_cache(); torch.cuda.synchronize()
torch.cuda.reset_peak_memory_stats()
with torch.no_grad():
    _ = model(x)
torch.cuda.synchronize()
inf_peak = peak_mb()
inf_resident = mem_mb()
print(f"Inference peak: {inf_peak:.2f} MB | Resident after: {inf_resident:.2f} MB")

# ---------- (B) Forward with grad: measure activation cost ----------
torch.cuda.empty_cache(); torch.cuda.synchronize()
torch.cuda.reset_peak_memory_stats()
out = model(x)
loss = criterion(out, y)
torch.cuda.synchronize()
train_peak = peak_mb()
train_resident = mem_mb()
saved_activations_peak = train_peak - inf_peak
print(f"Saved activations (forward with grad minus no-grad peak): {saved_activations_peak:.2f} MB")

# ---------- (C) Backward extra over forward-with-grad ----------
torch.cuda.reset_peak_memory_stats()
loss.backward()
torch.cuda.synchronize()
bwd_peak = peak_mb()
bwd_extra = bwd_peak - train_resident
print(f"Backward extra over forward-with-grad (grads + temps): {bwd_extra:.2f} MB")
print(f"Absolute peak during backward: {bwd_peak:.2f} MB")

# ---------- (D) Optimizer step ----------
torch.cuda.reset_peak_memory_stats()
optimizer.step()
torch.cuda.synchronize()
opt_step_peak = peak_mb()
print(f"Optimizer.step() extra (peak over post-backward): {opt_step_peak:.2f} MB")
optimizer.zero_grad(set_to_none=True)

# ---------- (E) Optional: backward temporaries only (preallocated grads) ----------
optimizer.zero_grad(set_to_none=False)
out = model(x); loss = criterion(out, y)
loss.backward()
with torch.no_grad():
    for p in model.parameters():
        if p.grad is not None:
            p.grad.zero_()
torch.cuda.synchronize()

out = model(x); loss = criterion(out, y)
torch.cuda.synchronize()
baseline_prealloc = mem_mb()
torch.cuda.reset_peak_memory_stats()
loss.backward()
torch.cuda.synchronize()
temps_only = peak_mb() - baseline_prealloc
print(f"(Optional) Backward temporaries only (preallocated grads): {temps_only:.2f} MB")

# --- Theoretical components (model-aware) ---
param_MB = (n_params * itemsize) / 1e6
grad_MB  = param_MB                        # one .grad tensor per parameter
adam_MB  = 2 * param_MB                    # Adam's m and v

# Very rough activation footprint for RNN-ish model (hidden state only)
acts_MB  = (B
            * RNN_params["N_neurons"]
            * RNN_params["N_layers"]
            * getattr(model, "max_steps", 1)
            * itemsize) / 1e6

theoretical_train_MB_no_overhead = param_MB + grad_MB + adam_MB + acts_MB
print(f"\n[Theory] params: {param_MB:.3f} MB, grads: {grad_MB:.3f} MB, "
      f"adam: {adam_MB:.3f} MB, activations≈{acts_MB:.3f} MB")
print(f"[Theory] training total (no CUDA/framework overhead): "
      f"{theoretical_train_MB_no_overhead:.3f} MB")

print(acts_MB)

