import numpy as np
import sys
sys.path.append('/home/anas/Desktop/Simulations/Training_NLO/nonlinear_oscillator_network/Utils')

import torch, gc, psutil, os


class CMAESOpt:
    """
    Full-covariance CMA-ES (PyTorch/CUDA) matching your numpy CMA_opt structure.
    WARNING: O(N^2) memory (C, B, invsqrtC).
    """

    def __init__(self, N_dim, pop_size, select_pop, sigma_init, mean_init,
                 device=None, dtype=torch.float32):
        self.N_dim = int(N_dim)
        self.pop_size = int(pop_size)
        self.select_pop = int(select_pop)

        self.device = torch.device(device) if device is not None else torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.dtype = dtype

        # mean init -> xmean (N,1)
        if isinstance(mean_init, np.ndarray):
            mean_init = torch.from_numpy(mean_init)
        elif not isinstance(mean_init, torch.Tensor):
            raise ValueError("mean_init must be a numpy array or torch.Tensor")

        if mean_init.ndim == 1:
            if mean_init.numel() != self.N_dim:
                raise ValueError(f"mean_init must have {self.N_dim} elements")
            self.xmean = mean_init.reshape(self.N_dim, 1).to(self.device, self.dtype)
        elif mean_init.ndim == 2:
            if mean_init.shape not in [(self.N_dim, 1), (1, self.N_dim)]:
                raise ValueError(f"mean_init must be of shape ({self.N_dim}, 1) or (1, {self.N_dim})")
            self.xmean = mean_init.reshape(self.N_dim, 1).to(self.device, self.dtype)
        else:
            raise ValueError("mean_init must be 1D or 2D")

        self.sigma = float(sigma_init)

        # weights (log scheme)
        w = torch.tensor(
            [np.log(self.select_pop + 0.5) - np.log(i) for i in range(1, self.select_pop + 1)],
            device=self.device, dtype=self.dtype
        )
        self.weights = w / w.sum()
        self.mueff = (self.weights.sum() ** 2) / (self.weights.pow(2).sum())

        # strategy params
        N = self.N_dim
        mueff = float(self.mueff.item())

        self.cc = (4 + mueff / N) / (N + 4 + 2 * mueff / N)
        self.cs = (mueff + 2) / (N + mueff + 5)
        self.c1 = 2 / ((N + 1.3) ** 2 + mueff)
        self.cmu = min(1 - self.c1, 2 * (mueff - 2 + 1 / mueff) / ((N + 2) ** 2 + mueff))
        self.damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (N + 1)) - 1) + self.cs

        # paths + covariance
        self.pc = torch.zeros((N, 1), device=self.device, dtype=self.dtype)
        self.ps = torch.zeros((N, 1), device=self.device, dtype=self.dtype)

        self.B = torch.eye(N, device=self.device, dtype=self.dtype)
        self.D = torch.ones((N, 1), device=self.device, dtype=self.dtype)
        self.C = self.B @ torch.diag((self.D.pow(2)).flatten()) @ self.B.T
        self.invsqrtC = self.B @ torch.diag((self.D.pow(-1)).flatten()) @ self.B.T

        self.eigeneval = 0
        self.chiN = (N ** 0.5) * (1 - 1 / (4 * N) + 1 / (21 * (N ** 2)))

        self.epsilon = 1e-8
        self.sigma_max = 1e10
        self.sigma_min = 1e-10

        self.eigen_update_frequency = int(N / 10) if int(N / 10) > 0 else 1

        self.population = None
        self.counteval = 0

        self.best_solution = None
        self.best_fitness = None

    @torch.no_grad()
    def ask(self) -> torch.Tensor:
        Z = torch.randn((self.N_dim, self.pop_size), device=self.device, dtype=self.dtype)
        BDZ = self.B @ (self.D * Z)  # (N, pop)
        self.population = self.xmean + (self.sigma * BDZ)  # (N, pop)
        return self.population.T  # (pop, N)

    @torch.no_grad()
    def tell(self, reward_table) -> None:
        # to tensor
        if isinstance(reward_table, (list, tuple, np.ndarray)):
            arfitness = torch.tensor(reward_table, device=self.device, dtype=self.dtype).flatten()
        elif isinstance(reward_table, torch.Tensor):
            arfitness = reward_table.to(self.device, self.dtype).flatten()
        else:
            raise ValueError("reward_table must be list, numpy array, or torch.Tensor")

        if arfitness.numel() != self.pop_size:
            raise ValueError("reward_table length must equal pop_size")

        # sort ascending (minimization)
        arindex = torch.argsort(arfitness, dim=0)
        xold = self.xmean.clone()

        # mean update
        best_pop = self.population[:, arindex[:self.select_pop]]  # (N, select)
        self.xmean = best_pop @ self.weights.reshape(self.select_pop, 1)

        # evolution path ps
        y = (self.xmean - xold) / (self.sigma + self.epsilon)
        self.ps = (1 - self.cs) * self.ps + (np.sqrt(self.cs * (2 - self.cs) * float(self.mueff.item()))) * (self.invsqrtC @ y)

        denom = (1 - (1 - self.cs) ** (2 * self.counteval / self.pop_size) + self.epsilon)
        hsig = (self.ps.pow(2).sum() / denom / self.N_dim) < (2 + 4 / (self.N_dim + 1))
        hsig_f = 1.0 if bool(hsig.item()) else 0.0

        # evolution path pc
        self.pc = (1 - self.cc) * self.pc + (hsig_f * np.sqrt(self.cc * (2 - self.cc) * float(self.mueff.item()))) * y

        # covariance update
        artmp = (best_pop - xold.repeat(1, self.select_pop)) / (self.sigma + self.epsilon)  # (N, select)

        C_new = (1 - self.c1 - self.cmu) * self.C
        C_rank1 = self.c1 * (self.pc @ self.pc.T + (1 - hsig_f) * self.cc * (2 - self.cc) * self.C)
        C_rankmu = self.cmu * (artmp @ torch.diag(self.weights) @ artmp.T)

        self.C = C_new + C_rank1 + C_rankmu
        self.C = self.C + (self.epsilon * torch.eye(self.N_dim, device=self.device, dtype=self.dtype))

        # sigma update
        ps_norm = torch.linalg.norm(self.ps).item()
        self.sigma *= float(np.exp((self.cs / self.damps) * (ps_norm / self.chiN - 1)))
        self.sigma = min(max(self.sigma, self.sigma_min), self.sigma_max)

        # eigen update occasionally
        if (self.counteval % self.eigen_update_frequency) == 0:
            self.C = 0.5 * (self.C + self.C.T)
            evals, evecs = torch.linalg.eigh(self.C)
            evals = torch.clamp(evals, min=self.epsilon)
            self.B = evecs
            self.D = torch.sqrt(evals).reshape(self.N_dim, 1)

            invD = (1.0 / (self.D + self.epsilon)).flatten()
            self.invsqrtC = self.B @ torch.diag(invD) @ self.B.T

        # best tracking
        current_best_index = torch.argmin(arfitness).item()
        current_best_solution = self.population[:, current_best_index]
        current_best_fitness = float(arfitness[current_best_index].item())

        if (self.best_fitness is None) or (current_best_fitness < self.best_fitness):
            self.best_solution = current_best_solution.clone()
            self.best_fitness = current_best_fitness

        self.counteval += 1


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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
n_loops = 5
warmup_loops = 5


# IMPORTANT: full CMA-ES O(N^2). Be careful with big numbers here added a try catch to see if mem is enough
N_dim_vec = [   50,   200,   800,  1800,  5000, 11250]

# Dummy reward values (constant)
dummy_reward_value = 1.0

# Logging arrays
mem_before_MB = []
mem_after_MB  = []
peak_memory_MB = []
delta_after_before_MB = []
update_cost_MB = []
status = []  # "ok" or "oom"

last_dtype = None


# ==========================================================
# Sweep Loop
# ==========================================================
for N_dim in N_dim_vec:
    print("\n" + "="*60)
    print(f"CMA-ES memory test - Parameter count: {N_dim}")
    print("="*60)

    # Cleanup
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()

    try:
        # ======================================================
        # Init
        # ======================================================
        init_pos = torch.randn(N_dim, 1, dtype=dtype, device=device)

        pop_size = max(int(0.01 * N_dim), 10)
        select_pop = max(pop_size // 2, 2)

        cma = CMAESOpt(
            N_dim=N_dim,
            pop_size=pop_size,
            select_pop=select_pop,
            sigma_init=1e-1,
            mean_init=init_pos,   # (N,1)
            device=device,
            dtype=dtype,
        )

        rewards = torch.full((pop_size, 1), dummy_reward_value, dtype=dtype, device=device)

        # ======================================================
        # Warm-up (not measured)
        # ======================================================
        for _ in range(warmup_loops):
            candidates = cma.ask()
            cma.tell(rewards)

        # ======================================================
        # Memory measurement
        # ======================================================
        if device.type == "cuda":
            reset_peak()
            m_before = mem_mb()
        else:
            # fallback if you ever run on CPU
            m_before = 0.0

        for _ in range(n_loops):
            candidates = cma.ask()
            cma.tell(rewards)

        if device.type == "cuda":
            m_after = mem_mb()
            m_peak = peak_mb()
        else:
            m_after, m_peak = 0.0, 0.0

        last_dtype = candidates.dtype

        mem_before_MB.append(m_before)
        mem_after_MB.append(m_after)
        peak_memory_MB.append(m_peak)
        delta_after_before_MB.append(m_after - m_before)
        update_cost_MB.append(m_peak - m_before)
        status.append("ok")

        print(f"pop_size={pop_size}, select_pop={select_pop}")
        print(f"Peak memory: {m_peak:.3f} MB")

    except RuntimeError as e:
        # catch CUDA OOM, mark as failed, keep sweep going
        if "out of memory" in str(e).lower():
            print("CUDA OOM at N_dim =", N_dim)
            mem_before_MB.append(float("nan"))
            mem_after_MB.append(float("nan"))
            peak_memory_MB.append(float("nan"))
            delta_after_before_MB.append(float("nan"))
            update_cost_MB.append(float("nan"))
            status.append("oom")
            # cleanup after OOM
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            gc.collect()
        else:
            raise


# ==========================================================
# Final Summary
# ==========================================================
print("\n================================")
print("CMA-ES Memory Sweep Results Summary")
print("================================")
print("N_dim_vec =", N_dim_vec)
print("status   =", status)

print("\n-- Memory BEFORE updates (MB) --")
print([None if (isinstance(v, float) and (v != v)) else round(v, 3) for v in mem_before_MB])

print("\n-- Memory AFTER updates (MB) --")
print([None if (isinstance(v, float) and (v != v)) else round(v, 3) for v in mem_after_MB])

print("\n-- PEAK memory (MB) [main metric] --")
print([None if (isinstance(v, float) and (v != v)) else round(v, 3) for v in peak_memory_MB])

print("\n-- Delta (after - before) (MB) --")
print([None if (isinstance(v, float) and (v != v)) else round(v, 3) for v in delta_after_before_MB])

print("\n-- Update Cost (peak - before) (MB) --")
print([None if (isinstance(v, float) and (v != v)) else round(v, 3) for v in update_cost_MB])

print("\ndtype of candidates from last ok run:", last_dtype)
print("======================")
