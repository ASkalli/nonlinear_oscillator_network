import math

def estimate_pepg_mem_MB(N_params,
                         pop_ratio=0.01,
                         min_pop=10,
                         dtype_bytes=4,
                         keep_best=True,
                         include_candidates=True,
                         include_sigma_update=True):
    """
    Theoretical memory estimate (in MB) for one PEPG ask+tell update,
    counting PEPG-related tensors (mu, sigma, noise, candidates, etc.).

    N_params:     number of parameters in the model.
    pop_ratio:    population size as a fraction of N_params (e.g. 0.01 → 1%).
    min_pop:      minimum population size.
    dtype_bytes:  bytes per element (4 for float32).
    """

    # ---- Population size (PEPG enforces even population) ----
    pop_raw  = max(int(pop_ratio * N_params), min_pop)
    pop_even = pop_raw if (pop_raw % 2 == 0) else pop_raw + 1
    batch    = pop_even // 2  # antithetic sampling: epsilon has 'batch' rows

    # =========================================================
    # 1) Persistent state (always resident in the optimizer)
    # =========================================================
    # mu:        (N_params,)
    # sigma:     (N_params,)
    # best_mu:   (N_params,)  if keep_best=True
    n_mu_sigma = 2 * N_params
    n_best_mu  = N_params if keep_best else 0
    N_persistent = n_mu_sigma + n_best_mu

    # =========================================================
    # 2) Population-related tensors (per update)
    # =========================================================
    # epsilon:      (batch,   N_params)
    # epsilon_full: (pop_even, N_params)
    # candidates:   (pop_even, N_params)  -> returned by ask(), used for forward evals
    N_epsilon      = batch * N_params
    N_epsilon_full = pop_even * N_params
    N_candidates   = pop_even * N_params if include_candidates else 0

    N_population = N_epsilon + N_epsilon_full + N_candidates

    # =========================================================
    # 3) Rewards + ranking vectors
    # =========================================================
    # reward_table_result: (pop_even,)
    # reward_table:        (pop_even,)
    # ranks, order, rt, reward_avg, etc.
    # Rough upper bound: ~8 * pop_even elements total.
    N_rewards_ranks = 8 * pop_even

    # =========================================================
    # 4) Sigma-update internals
    # =========================================================
    # S:           (batch, N_params)
    # delta_sigma: (N_params,)
    # change_sigma:(N_params,)
    # max_change:  (N_params,)
    if include_sigma_update:
        N_S             = batch * N_params
        N_sigma_vectors = 3 * N_params
        N_sigma_update  = N_S + N_sigma_vectors
    else:
        N_sigma_update  = 0

    # =========================================================
    # Totals
    # =========================================================
    sections_elems = {
        "persistent_state": N_persistent,
        "population_tensors": N_population,
        "rewards_ranks": N_rewards_ranks,
        "sigma_update": N_sigma_update,
    }

    total_elems = sum(sections_elems.values())
    to_MB = lambda elems: round(elems * dtype_bytes / (1024**2), 3)

    sections_MB = {k: to_MB(v) for k, v in sections_elems.items()}
    sections_MB["total_MB"] = to_MB(total_elems)
    sections_MB["pop_even"] = pop_even
    sections_MB["batch"]    = batch

    return sections_MB



def count_params(N_in, N_h, N_out):
    # same as your BP script: 3 hidden layers
    W = N_in * N_h + 2 * N_h * N_h + N_h * N_out
    b = 3 * N_h + N_out
    return W + b


if __name__ == "__main__":

    N_neurons_vec = [5, 10, 20, 30, 50, 75, 100, 150, 180]
    N_dim_vec = [   50,   200,   800,  1800,  5000, 11250]
    N_dim_vec = [50,200,800,1100,1500,4096] # for QNN

    mem_cost_pepg = []
    n_params_vec  = []
    pops          = []

    #for N_h in N_neurons_vec:
    for N_params in N_dim_vec:
        #N_in, N_out = 784, 10
        #N_params = count_params(N_in, N_h, N_out)
        #n_params_vec.append(N_params)

        # PEPG: pop_size = 1% of N_params (min 10), enforced even
        out = estimate_pepg_mem_MB(
            N_params=N_params,
            pop_ratio=0.01,
            min_pop=10,
            dtype_bytes=4,
            keep_best=True,
            include_candidates=True,
            include_sigma_update=True,
        )

        mem_cost_pepg.append(out["total_MB"])
        pops.append(out["pop_even"])

    print("N_params:", n_params_vec)
    print("PEPG pop sizes:", pops)
    print("PEPG mem_cost (MB):", mem_cost_pepg)
