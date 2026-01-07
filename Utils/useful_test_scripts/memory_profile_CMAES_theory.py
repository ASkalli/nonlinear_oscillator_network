import math

def estimate_cmaes_mem_MB(N_dim,
                          pop_ratio=0.01,
                          min_pop=10,
                          select_ratio=0.5,
                          min_select=2,
                          dtype_bytes=4,
                          keep_best=True,
                          include_candidates=True,
                          include_population=True,
                          include_eigendecomp=True,
                          include_tell_temporaries=True):
    """
    Theoretical memory estimate (in MB) for one CMA-ES ask+tell update,
    counting CMA-ES-related tensors in the *full covariance* variant:
      xmean, pc, ps, B, D, C, invsqrtC, population, and key temporaries.

    N_dim:        number of parameters.
    pop_ratio:    pop_size as fraction of N_dim (like your PEPG script).
    select_ratio: select_pop as fraction of pop_size.
    dtype_bytes:  bytes per element (4 for float32).
    """

    # ---- Population / selection sizes ----
    pop_size = max(int(pop_ratio * N_dim), min_pop)
    select_pop = max(int(select_ratio * pop_size), min_select)
    select_pop = min(select_pop, pop_size)

    # =========================================================
    # 1) Persistent state (always resident)
    # =========================================================
    # xmean:     (N,1)
    # pc, ps:    (N,1)
    # weights:   (select_pop,)
    # D:         (N,1)
    # sigma:     scalar (ignore)
    N_persistent_vec = 4 * N_dim + select_pop  # xmean,pc,ps,D + weights

    # Full-covariance matrices (dominant):
    # B:         (N,N)
    # C:         (N,N)
    # invsqrtC:  (N,N)
    if include_eigendecomp:
        N_persistent_mats = 3 * (N_dim * N_dim)
    else:
        # if you disable eigendecomp terms, you're effectively estimating diag/separable CMA
        N_persistent_mats = 0

    # Best tracking (best_solution: (N,), best_fitness scalar)
    N_best = N_dim if keep_best else 0

    N_persistent = N_persistent_vec + N_persistent_mats + N_best

    # =========================================================
    # 2) ask() population generation tensors
    # =========================================================
    # Z:          (N, pop)
    # BDZ:        (N, pop)   [B @ (D*Z)]
    # population: (N, pop)   (stored)  [optional]
    # candidates: (pop, N)   returned view/copy (count as same size) [optional]
    N_Z   = N_dim * pop_size
    N_BDZ = N_dim * pop_size
    N_pop = (N_dim * pop_size) if include_population else 0
    N_cand = (N_dim * pop_size) if include_candidates else 0

    N_ask = N_Z + N_BDZ + N_pop + N_cand

    # =========================================================
    # 3) tell() internals (rough upper bound)
    # =========================================================
    # arfitness: (pop,)
    # arindex:   (pop,)   (int64 in torch, but we count as dtype_bytes for simplicity)
    # xold:      (N,1)
    # best_pop:  (N, select)
    # y:         (N,1)
    # artmp:     (N, select)
    #
    # Updates create some additional N×N temporaries (matmul results) briefly:
    # pc@pc.T:                (N,N)
    # artmp @ diag(w) @ artmp.T: (N,N)
    # plus a couple of intermediate matrices.
    # We'll count ~3*(N,N) as a safe upper bound for the update step.
    if include_tell_temporaries:
        N_tell_vecs = (2 * pop_size) + (3 * N_dim) + (2 * N_dim * select_pop)
        N_tell_mats = (3 * (N_dim * N_dim)) if include_eigendecomp else 0
        N_tell = N_tell_vecs + N_tell_mats
    else:
        N_tell = 0

    # =========================================================
    # Totals
    # =========================================================
    sections_elems = {
        "persistent_state": N_persistent,
        "ask_tensors": N_ask,
        "tell_temporaries": N_tell,
    }

    total_elems = sum(sections_elems.values())
    to_MB = lambda elems: round(elems * dtype_bytes / (1024**2), 3)

    sections_MB = {k: to_MB(v) for k, v in sections_elems.items()}
    sections_MB["total_MB"] = to_MB(total_elems)
    sections_MB["pop_size"] = pop_size
    sections_MB["select_pop"] = select_pop

    return sections_MB


def count_params(N_in, N_h, N_out):
    # same as your BP script: 3 hidden layers
    W = N_in * N_h + 2 * N_h * N_h + N_h * N_out
    b = 3 * N_h + N_out
    return W + b


if __name__ == "__main__":

    N_neurons_vec = [5, 10, 20, 30, 50, 75, 100, 150, 180]

    N_dim_vec = [   50,   200,   800,  1800,  5000, 11250]

    mem_cost_cma = []
    n_params_vec  = []
    pops          = []
    selects       = []

    #for N_h in N_neurons_vec:
    for N_params in N_dim_vec:
        #N_in, N_out = 784, 10
        #N_params = count_params(N_in, N_h, N_out)
        #n_params_vec.append(N_params)
        
        out = estimate_cmaes_mem_MB(
            N_dim=N_params,
            pop_ratio=0.01,
            min_pop=10,
            select_ratio=0.5,
            min_select=2,
            dtype_bytes=4,
            keep_best=True,
            include_candidates=True,
            include_population=True,
            include_eigendecomp=True,        # FULL CMA-ES (N×N)
            include_tell_temporaries=True,
        )

        mem_cost_cma.append(out["total_MB"])
        pops.append(out["pop_size"])
        selects.append(out["select_pop"])

    print("N_params:", n_params_vec)
    print("CMA pop sizes:", pops)
    print("CMA select sizes:", selects)
    print("CMA mem_cost (MB):", mem_cost_cma)
