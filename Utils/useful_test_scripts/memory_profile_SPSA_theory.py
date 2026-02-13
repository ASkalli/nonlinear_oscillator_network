import math

def estimate_spsa_mem_MB(N_params,
                        dtype_bytes=4,
                        include_adam_helpers=True):
    """
    Theoretical memory estimate (in MB) for one SPSA + Adam update,
    counting only SPSA/Adam-related tensors (params, perturbations,
    gradient estimate, Adam helper vectors, etc.).

    N_params:        number of parameters in the model.
    dtype_bytes:     bytes per element (4 for float32).
    include_adam_helpers:
        - If True, includes Adam's moving-average vectors m and v,
        each of size (N_params,).
    """

    # =========================================================
    # 1) Persistent state (always resident in the optimizer)
    # =========================================================
    # SPSA.params: (N_params,)
    N_persistent = N_params

    # Adam helper vectors (moving averages):
    # m: (N_params,)
    # v: (N_params,)
    if include_adam_helpers:
        N_adam_helpers = 2 * N_params
    else:
        N_adam_helpers = 0

    # =========================================================
    # 2) Perturbation-related tensors (per update)
    # =========================================================
    # In perturb_parameters():
    #   rnd:          (N_params,)
    #   delta:        (N_params,)
    #   params_plus:  (N_params,)
    #   params_minus: (N_params,)
    #
    # In practice some of these can be reused/freed early, but as an
    # upper-bound we count 4 * N_params elements.
    N_perturb_tensors = 4 * N_params

    # =========================================================
    # 3) Gradient and step tensors (per update)
    # =========================================================
    # approximate_gradient():
    #   grad: (N_params,)
    #
    # AdamOpt_torch.step():
    #   step: (N_params,)  [the scaled update returned by Adam]
    #
    # Again, these could be reused, but we count them separately.
    N_grad_step = 2 * N_params

    # =========================================================
    # Totals
    # =========================================================
    sections_elems = {
        "persistent_state": N_persistent,
        "perturbation_tensors": N_perturb_tensors,
        "grad_and_step": N_grad_step,
        "adam_helpers": N_adam_helpers,
    }

    total_elems = sum(sections_elems.values())
    to_MB = lambda elems: round(elems * dtype_bytes / (1024**2), 3)

    sections_MB = {k: to_MB(v) for k, v in sections_elems.items()}
    sections_MB["total_MB"] = to_MB(total_elems)
    sections_MB["N_params"] = N_params

    return sections_MB


def count_params(N_in, N_h, N_out):
    # same as your BP / PEPG script: 3 hidden layers
    W = N_in * N_h + 2 * N_h * N_h + N_h * N_out
    b = 3 * N_h + N_out
    return W + b


if __name__ == "__main__":

    N_neurons_vec = [5, 10, 20 ,30, 50, 75, 100,150,250]
    N_dim_vec = [   50,   200,   800,  1800,  5000, 11250]
    N_dim_vec = [50,200,800,1100,1500,4096] # for QNN

    mem_cost_spsa = []  
    n_params_vec  = []

    #for N_h in N_neurons_vec:
    for N_params in N_dim_vec:
        #N_in, N_out = 784, 10
        #N_params = count_params(N_in, N_h, N_out)
        #n_params_vec.append(N_params)


        out = estimate_spsa_mem_MB(
            N_params=N_params,
            dtype_bytes=4,
            include_adam_helpers=True,   # count Adam m and v
        )

        mem_cost_spsa.append(out["total_MB"])

    print("N_params:", n_params_vec)
    print("SPSA mem_cost (MB):", mem_cost_spsa)
