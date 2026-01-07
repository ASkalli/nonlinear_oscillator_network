import numpy as np
import sys
sys.path.append('/home/anas/Desktop/Simulations/Training_NLO/nonlinear_oscillator_network/Utils')

import torch, gc, psutil, os


class PEPGOpt:
    def __init__(self, num_params, pop_size, learning_rate, starting_mu, starting_sigma,
                 device=None, dtype=torch.float32):
        self.pop_size = pop_size + 1 if pop_size % 2 else pop_size
        self.num_params = num_params
        self.device = torch.device(device) if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = dtype

        self.mu = (starting_mu if isinstance(starting_mu, torch.Tensor)
                   else torch.tensor(starting_mu)).flatten().to(self.device, self.dtype)
        if self.mu.numel() != num_params:
            raise ValueError(f"starting_mu must have {num_params} elements")

        self.batch_size = self.pop_size // 2
        self.sigma_init = float(starting_sigma)
        self.sigma = torch.full((num_params,), self.sigma_init, device=self.device, dtype=self.dtype)

        # Hyperparameters
        self.sigma_alpha = 0.30
        self.sigma_decay = 0.999
        self.sigma_limit = 0.01
        self.sigma_max_change = 0.2
        self.learning_rate = float(learning_rate)
        self.learning_rate_decay = 0.99
        self.learning_rate_limit = 0.001
        self.elite_ratio = 0.0
        self.weight_decay = 0.01
        self.forget_best = True
        self.rank_fitness = True
        self.average_baseline = True
        self.use_elite = False

        self.first_iteration = True
        self.best_reward = None
        self.best_mu = None

        # state placeholders
        self.epsilon = None
        self.epsilon_full = None
        self.curr_best_reward = None
        self.curr_best_mu = None

    def compute_ranks(self, x: torch.Tensor) -> torch.Tensor:
        x = x.flatten()
        order = torch.argsort(x)  # ascending
        ranks = torch.empty_like(x, dtype=torch.float32)
        ranks[order] = torch.arange(x.numel(), device=x.device, dtype=torch.float32)
        return ranks

    def compute_centered_ranks(self, x: torch.Tensor) -> torch.Tensor:
        r = self.compute_ranks(x)
        return r / (x.numel() - 1) - 0.5

    @torch.no_grad()
    def ask(self) -> torch.Tensor:
        # epsilon ~ N(0, sigma^2), antithetic sampling
        self.epsilon = torch.randn(self.batch_size, self.num_params, device=self.device, dtype=self.dtype) * self.sigma
        self.epsilon_full = torch.cat([self.epsilon, -self.epsilon], dim=0)
        if self.average_baseline:
            # shape: (pop_size, num_params)
            return self.mu.unsqueeze(0) + self.epsilon_full
        else:
            zeros = torch.zeros(1, self.num_params, device=self.device, dtype=self.dtype)
            return torch.cat([zeros, self.mu.unsqueeze(0) + self.epsilon_full], dim=0)

    @torch.no_grad()
    def tell(self, reward_table_result) -> int:
        reward_table = (reward_table_result if isinstance(reward_table_result, torch.Tensor)
                        else torch.tensor(reward_table_result, device=self.device, dtype=self.dtype)).flatten()

        if self.rank_fitness:
            reward_table = self.compute_centered_ranks(reward_table)

        b = reward_table.mean() if self.average_baseline else reward_table[0]
        reward = reward_table if self.average_baseline else reward_table[1:]

        # choose best by minimum (as in your original code)
        best_idx = torch.argmin(reward)
        best_reward = reward[best_idx]

        if (best_reward > b) or self.average_baseline:
            best_mu = self.mu + self.epsilon_full[best_idx]
        else:
            best_mu = self.mu.clone()

        self.curr_best_reward = best_reward
        self.curr_best_mu = best_mu

        if self.first_iteration:
            self.sigma.fill_(self.sigma_init)
            self.first_iteration = False
            self.best_reward = best_reward.clone()
            self.best_mu = best_mu.clone()
        elif self.forget_best or (best_reward > self.best_reward):
            self.best_reward = best_reward.clone()
            self.best_mu = best_mu.clone()

        # ---- Update mu ----
        if self.use_elite and self.elite_ratio > 0:
            k = max(1, int(self.elite_ratio * reward_table.numel()))
            elite_idx = torch.argsort(reward_table)[:k]
            self.mu += self.epsilon_full[elite_idx].mean(dim=0)
        else:
            rt = reward_table[:self.batch_size] - reward_table[self.batch_size:]  # (batch,)
            change_mu = rt @ self.epsilon_full[:self.batch_size]                  # (num_params,)
            self.mu -= self.learning_rate * change_mu

        # ---- Update sigma ----
        if self.sigma_alpha > 0:
            if self.rank_fitness:
                stdev_reward = 1.0
            else:
                stdev_reward = reward_table_result.std().item() if isinstance(reward_table_result, torch.Tensor) \
                               else torch.tensor(reward_table_result, device=self.device, dtype=self.dtype).std().item()
                stdev_reward = max(stdev_reward, 1e-12)

            S = (self.epsilon**2 - self.sigma**2) / (self.sigma + 1e-12)         # (batch, num_params)
            reward_avg = 0.5 * (reward_table[:self.batch_size] + reward_table[self.batch_size:])  # (batch,)
            rS = reward_avg - b                                                   # (batch,)
            delta_sigma = (rS.unsqueeze(0) @ S).squeeze(0) / (2 * self.batch_size * stdev_reward)  # (num_params,)

            change_sigma = self.sigma_alpha * delta_sigma
            max_change = self.sigma_max_change * self.sigma
            change_sigma = torch.clamp(change_sigma, -max_change, max_change)

            self.sigma -= change_sigma
            # decay & floor
            if self.sigma_decay < 1.0:
                self.sigma.mul_(self.sigma_decay)
            self.sigma = torch.maximum(self.sigma, torch.full_like(self.sigma, self.sigma_limit))

        # ---- learning rate decay ----
        if (self.learning_rate_decay < 1.0) and (self.learning_rate > self.learning_rate_limit):
            self.learning_rate *= self.learning_rate_decay

        return 0

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
#N_dim_vec = [4045, 8180, 16750, 25720, 44860, 71035, 99710, 164560, 208270]

#experiment 
#N_dim_vec = [  50,   200,   800,  1800,  5000, 11250]

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
print("======================")
