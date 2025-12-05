import torch
import numpy as np
import sys
sys.path.append('/home/anas/Desktop/Simulations/Training_NLO/nonlinear_oscillator_network/Utils')

import tracemalloc
import psutil, os   
import scipy
import time
from optimization_algorithms import *
import gc

import sys, psutil, os, gc, tracemalloc, time


class SPSAOpt_torch:
    def __init__(self, params, alpha=1e-2, epsilon=1e-5, device=None, dtype=torch.float32):
        if isinstance(params, torch.Tensor):
            self.params = params.detach().clone().to(device=device, dtype=dtype)
        else:
            self.params = torch.tensor(params, device=device, dtype=dtype)
        self.alpha = alpha
        self.epsilon = epsilon
        self.iteration = 0
        self.delta = None

    def perturb_parameters(self):
        """Generate {-1, +1} perturbation vector and return plus/minus params."""
        rnd = torch.rand_like(self.params)
        self.delta = torch.where(rnd > 0.5, torch.ones_like(self.params), -torch.ones_like(self.params))
        params_plus = self.params + self.epsilon * self.delta
        params_minus = self.params - self.epsilon * self.delta
        return params_plus, params_minus

    def approximate_gradient_func(self, loss_func):
        """Approximate gradient using a provided loss function callable."""
        # assumes self.delta was set by perturb_parameters()
        params_plus = self.params + self.epsilon * self.delta
        params_minus = self.params - self.epsilon * self.delta
        loss_plus = loss_func(params_plus)
        loss_minus = loss_func(params_minus)
        var_delta = torch.var(self.delta.float(), unbiased=False)
        grad = ((loss_plus - loss_minus) / (2 * self.epsilon * (var_delta + 1e-12))) * self.delta
        return grad

    def approximate_gradient(self, loss_plus, loss_minus):
        """Approximate gradient from precomputed loss values."""
        var_delta = torch.var(self.delta.float(), unbiased=False)
        denom = 2 * self.epsilon * (var_delta + 1e-12)
        # if var is exactly zero, fall back to epsilon-only denominator (kept from your numpy version)
        if var_delta.item() == 0.0:
            denom = 2 * self.epsilon + 1e-7
        grad = ((loss_plus - loss_minus) / denom) * self.delta
        return grad

    def update_parameters(self, gradient):
        """params <- params - alpha * gradient"""
        with torch.no_grad():
            self.params -= self.alpha * gradient
        return self.params

    def update_parameters_step(self, step):
        """params <- params - step (step is a vector)"""
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
        """Return the Adam step (vector to subtract from params)."""
        self.iteration += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grad ** 2)

        m_hat = self.m / (1 - self.beta1 ** self.iteration)
        v_hat = self.v / (1 - self.beta2 ** self.iteration)

        step = self.lr * m_hat / (torch.sqrt(v_hat) + self.epsilon)
        return step



# ---- helper functions ----
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

def rss_mb():
    return psutil.Process(os.getpid()).memory_info().rss / 1e6


# ---- config ----
N_dim = 324260
dtype = torch.float32
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
n_loops = 5
warmup_loops = 5
# ---- init ----
init_pos = torch.randn(N_dim, 1, device=device, dtype=dtype)
spsa_optimizer = SPSAOpt_torch(init_pos, alpha=3e-4, epsilon=1e-5, device=device, dtype=dtype)
adam_optimizer = AdamOpt_torch(init_pos, lr=3e-4, beta1=0.9, beta2=0.9, epsilon=1e-8, device=device, dtype=dtype)

reward_plus  = 0.4426910220384598
reward_minus = 0.44264774668216705

# warmup (not measured)
for _ in range(warmup_loops):
    params_plus, params_minus = spsa_optimizer.perturb_parameters()
    grad_spsa = spsa_optimizer.approximate_gradient(reward_plus, reward_minus)
    step = adam_optimizer.step(grad_spsa)
    current_params = spsa_optimizer.update_parameters_step(step)


# ---- memory measurement ----
if device.type == 'cuda':
    reset_peak()
    mem_before = mem_mb()
else:
    mem_before = rss_mb()


for _ in range(n_loops):
    params_plus, params_minus = spsa_optimizer.perturb_parameters()
    grad_spsa = spsa_optimizer.approximate_gradient(reward_plus, reward_minus)
    step = adam_optimizer.step(grad_spsa)
    current_params = spsa_optimizer.update_parameters_step(step)

if device.type == 'cuda':
    mem_after = mem_mb()
    peak = peak_mb()
else:
    mem_after = rss_mb()
    peak = mem_after  # no peak tracking on CPU

# ---- report ----
print(f"Device: {device}")
print(f"Memory before update: {mem_before:.3f} MB")
print(f"Memory after update:  {mem_after:.3f} MB")
print(f"Peak memory used:     {peak:.3f} MB")
print(f"Delta (peak - before): {peak - mem_before:.3f} MB")

print(params_plus.dtype)