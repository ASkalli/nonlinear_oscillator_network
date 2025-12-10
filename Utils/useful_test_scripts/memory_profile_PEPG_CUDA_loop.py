import numpy as np
import sys
sys.path.append('/home/anas/Desktop/Simulations/Training_NLO/nonlinear_oscillator_network/Utils')

import torch, gc, psutil, os
from optimization_algorithms import *

# ==========================================================
# Memory helpers
# ==========================================================
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


# ==========================================================
# Sweep configuration
# ==========================================================
dtype = torch.float32
device = torch.device('cuda')
n_loops = 5
warmup_loops = 5

# Provided parameter sizes (same as SPSA sweep)
N_dim_vec = [4045, 8180, 16750, 25720, 44860, 71035, 99710, 164560, 208270]

# Dummy reward values (constant)
dummy_reward_value = 1.0

# Logging arrays
mem_before_MB = []
mem_after_MB  = []
peak_memory_MB = []
delta_after_before_MB = []
update_cost_MB = []

last_dtype = None


# ==========================================================
# Sweep Loop
# ==========================================================
for N_dim in N_dim_vec:
    print("\n" + "="*60)
    print(f"PEPG memory test - Parameter count: {N_dim}")
    print("="*60)

    # Cleanup
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    gc.collect()

    # ======================================================
    # Init
    # ======================================================
    init_pos = torch.randn(N_dim, 1, dtype=dtype, device=device)

    pop_size = max(int(0.01 * N_dim), 10)
    pop_size = pop_size + 1 if pop_size % 2 else pop_size

    pepg = PEPGOpt(
        num_params=N_dim,
        pop_size=pop_size,
        learning_rate=0.01,
        starting_mu=init_pos.squeeze(1),
        starting_sigma=1e-1,
        device=device,
        dtype=dtype,
    )

    rewards = torch.full((pop_size, 1), dummy_reward_value, dtype=dtype, device=device)

    # ======================================================
    # Warm-up (not measured)
    # ======================================================
    for _ in range(warmup_loops):
        candidates = pepg.ask()
        pepg.tell(rewards)

    # ======================================================
    # Memory measurement
    # ======================================================
    reset_peak()
    m_before = mem_mb()

    for _ in range(n_loops):
        candidates = pepg.ask()
        pepg.tell(rewards)

    m_after = mem_mb()
    m_peak  = peak_mb()

    last_dtype = candidates.dtype

    # Store results
    mem_before_MB.append(m_before)
    mem_after_MB.append(m_after)
    peak_memory_MB.append(m_peak)
    delta_after_before_MB.append(m_after - m_before)
    update_cost_MB.append(m_peak - m_before)

    print(f"Peak memory: {m_peak:.3f} MB")


# ==========================================================
# Final Summary
# ==========================================================
print("\n================================")
print("PEPG Memory Sweep Results Summary")
print("================================")
print("N_dim_vec =", N_dim_vec)

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

print("\ndtype of candidates from last run:", last_dtype)
print("======================
