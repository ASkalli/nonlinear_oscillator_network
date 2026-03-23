import os, psutil, gc, sys, time
import torch
import torch.nn as nn
import torch.optim as optim

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
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

# ----------------------------------------------------------
# Env
# ----------------------------------------------------------
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
torch.set_num_threads(1)
assert torch.cuda.is_available(), "CUDA required"

dtype = torch.float32
device = torch.device("cuda")

# ----------------------------------------------------------
# Hyperparameters
# ----------------------------------------------------------
RNN_params = {"N_in": 784, "N_out": 10, "N_neurons": 400, "N_layers": 3}

B = 100
criterion = nn.CrossEntropyLoss()
n_loops = 5
warmup_loops = 5

# Neuron sweep target
N_neurons_vec_BP = [5, 10, 20, 30, 50, 75, 100, 150, 250, 300, 400]

# Arrays to store results
mem_before_MB = []
mem_after_MB = []
peak_memory_MB = []
delta_after_before_MB = []
update_cost_MB = []

# ----------------------------------------------------------
# Sweep
# ----------------------------------------------------------
for N_neurons in N_neurons_vec_BP:
    print("\n" + "="*50)
    print(f"Testing N_neurons = {N_neurons}")
    print("="*50)

    # Update model size
    RNN_params["N_neurons"] = N_neurons

    # Free old memory
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    gc.collect()

    # Build model
    model = simple_FFNN(params=RNN_params).to(device)
    optimizer = optim.Adam(model.parameters(), lr=3e-4)

    # Dummy data
    x = torch.randn(B, RNN_params["N_in"], dtype=dtype, device=device)
    y = torch.randint(0, RNN_params["N_out"], (B,), dtype=torch.long, device=device)

    # Warm-up to stabilize memory allocation
    for _ in range(warmup_loops):
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()

    torch.cuda.synchronize()

    # Measurement
    reset_peak()
    mbefore = mem_mb()
    for _ in range(n_loops):
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
    mafter = mem_mb()
    mpeak = peak_mb()

    # Store values
    mem_before_MB.append(mbefore)
    mem_after_MB.append(mafter)
    peak_memory_MB.append(mpeak)
    delta_after_before_MB.append(mafter - mbefore)
    update_cost_MB.append(mpeak - mbefore)

    print(f"Peak memory used: {mpeak:.3f} MB")

# ----------------------------------------------------------
# Final Summary
# ----------------------------------------------------------
print("\n================================")
print("Memory Sweep Results Summary")
print("================================")
print(f"N_neurons_vec_BP = {N_neurons_vec_BP}")

print("\n-- Memory BEFORE updates (MB) --")
print([round(v, 3) for v in mem_before_MB])

print("\n-- Memory AFTER updates (MB) --")
print([round(v, 3) for v in mem_after_MB])

print("\n-- PEAK memory (MB) [main metric] --")
print([round(v, 3) for v in peak_memory_MB])

print("\n-- Delta (after - before) (MB) --")
print([round(v, 3) for v in delta_after_before_MB])

print("\n-- Update Cost (peak - before) (MB) --")
print([round(v, 3) for v in update_cost_MB])
print("\n================================\n")
