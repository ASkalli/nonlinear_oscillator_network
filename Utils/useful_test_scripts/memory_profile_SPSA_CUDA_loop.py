import torch
import numpy as np
import sys
sys.path.append('/home/anas/Desktop/Simulations/Training_NLO/nonlinear_oscillator_network/Utils')

import psutil, os, gc, time
from optimization_algorithms import *

# ===========================
#  Optimizer classes (unchanged)
# ===========================
class SPSAOpt_torch:
    def __init__(self, params, alpha=1e-2, epsilon=1e-5, device=None, dtype=torch.float32):
        if isinstance(params, torch.Tensor):
            self.params = params.detach().clone().to(device=device, dtype=dtype)
        else:
            self.params = torch.tensor(params, device=device, dtype=dtype)
        self.alpha = alpha
        self.epsilon = epsilon
        self.delta = None

    def perturb_parameters(self):
        rnd = torch.rand_like(self.params)
        self.delta = torch.where(rnd > 0.5, torch.ones_like(self.params), -torch.ones_like(self.params))
        params_plus  = self.params + self.epsilon * self.delta
        params_minus = self.params - self.epsilon * self.delta
        return params_plus, params_minus

    def approximate_gradient(self, loss_plus, loss_minus):
        var_delta = torch.var(self.delta.float(), unbiased=False)
        denom = 2 * self.epsilon * (var_delta + 1e-12)
        if var_delta.item() == 0.0:
            denom = 2 * self.epsilon + 1e-7
        grad = ((loss_plus - loss_minus) / denom) * self.delta
        return grad

    def update_parameters_step(self, step):
        with torch.no_grad():
            self.params -= step
        return self.params


class AdamOpt_torch:
    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, epsilon=1e-8, device=None, dtype=torch.float32):
        if isinstance(params, torch.Tensor):
            self.params = params.detach().clone().to(device=device, dtype=dtype)
        else:
            self.params = torch.tensor(params, device=device, dtype=dtype)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = torch.zeros_like(self.params)
        self.v = torch.zeros_like(self.params)
        self.iteration = 0

    def step(self, grad):
        self.iteration += 1

        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grad ** 2)

        m_hat = self.m / (1 - self.beta1 ** self.iteration)
        v_hat = self.v / (1 - self.beta2 ** self.iteration)

        return self.lr * m_hat / (torch.sqrt(v_hat) + self.epsilon)


# ===========================
#  Memory helpers
# ===========================
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


# ===========================
# Sweep configuration
# ===========================
dtype = torch.float32
device = torch.device('cuda')
n_loops = 5
warmup_loops = 5

# Provided parameter sizes
N_dim_vec = [4045, 8180, 16750, 25720, 44860, 71035, 99710, 164560, 324260]

reward_plus  = 0.4426910220384598
reward_minus = 0.44264774668216705

# Memory logging arrays
mem_before_MB = []
mem_after_MB  = []
peak_memory_MB = []
delta_after_before_MB = []
update_cost_MB = []

last_dtype = None


# ===========================
# Sweep
# ===========================
for N_dim in N_dim_vec:
    print("\n" + "="*60)
    print(f"SPSA memory test - Parameter count: {N_dim}")
    print("="*60)

    # Cleanup
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    gc.collect()

    init_pos = torch.randn(N_dim, 1, device=device, dtype=dtype)
    spsa = SPSAOpt_torch(init_pos, alpha=3e-4, epsilon=1e-5,
                         device=device, dtype=dtype)
    adam = AdamOpt_torch(init_pos, lr=3e-4, beta1=0.9, beta2=0.9,
                         epsilon=1e-8, device=device, dtype=dtype)

    # Warm-up
    for _ in range(warmup_loops):
        params_plus, params_minus = spsa.perturb_parameters()
        grad = spsa.approximate_gradient(reward_plus, reward_minus)
        step = adam.step(grad)
        spsa.update_parameters_step(step)

    # Measurement
    reset_peak()
    m_before = mem_mb()

    for _ in range(n_loops):
        params_plus, params_minus = spsa.perturb_parameters()
        grad = spsa.approximate_gradient(reward_plus, reward_minus)
        step = adam.step(grad)
        spsa.update_parameters_step(step)

    m_after = mem_mb()
    m_peak  = peak_mb()

    # Store results
    mem_before_MB.append(m_before)
    mem_after_MB.append(m_after)
    peak_memory_MB.append(m_peak)
    delta_after_before_MB.append(m_after - m_before)
    update_cost_MB.append(m_peak - m_before)

    last_dtype = params_plus.dtype

    # Print short status
    print(f"Peak memory used: {m_peak:.3f} MB")


# ===========================
# Final arrays
# ===========================
print("\n================================")
print("SPSA Memory Sweep Results Summary")
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

print("\nparams_plus.dtype:", last_dtype)
print("================================\n")
