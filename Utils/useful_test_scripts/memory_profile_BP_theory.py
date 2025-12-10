
def estimate_mem_MB(N_in, N_h, N_out, B, dtype_bytes=4,
                    snapshots=True, cache_input=True, cache_logits=True):
    # ----- parameters -----
    W = N_in*N_h + 2*N_h*N_h + N_h*N_out
    b = 3*N_h + N_out
    N_params = W + b
    N_grads  = N_params

    # ----- forward cache -----
    acts = (cache_input * N_in) + (3*N_h) + (cache_logits * N_out)
    pre  = (3*N_h) + N_out
    N_fw = B * (acts + pre)  # a^0..a^3, logits + z^1..z^3, z_out

    # ----- backward cache -----
    dA = 3 * B * N_h
    dZ = B * (3*N_h + N_out)
    dX = B * N_in
    snaps_Wb = (W + b) if snapshots else 0
    N_bw = dA + dZ + dX + snaps_Wb

    # ----- totals -----
    sections = {
        "params": N_params,
        "grads": N_grads,
        "forward_cache": N_fw,
        "backward_cache": N_bw,
    }
    total_elems = sum(sections.values())
    to_MB = lambda elems: round(elems * dtype_bytes / (1024**2), 3)

    return {k: to_MB(v) for k, v in sections.items()} | {"total_MB": to_MB(total_elems)}


def count_params(N_in, N_h, N_out):
    # 3 hidden layers
    W = N_in*N_h + 2*N_h*N_h + N_h*N_out
    b = 3*N_h + N_out
    return W + b



if __name__ == "__main__":
    
    N_neurons_vec = [5, 10, 20 ,30, 50, 75, 100,150,250,300,400]

    mem_cost = []
    n_params_vec = []

    for k in range(len(N_neurons_vec)):

        N_in, N_h, N_out, B = 784, N_neurons_vec[k], 10, 10
        n_params = count_params(N_in, N_h, N_out)

        out = estimate_mem_MB(N_in, N_h, N_out, B, dtype_bytes=4,
                            snapshots=True, cache_input=True, cache_logits=True)

        mem_cost.append(out['total_MB'])
        n_params_vec.append(n_params)

    print(mem_cost)
    print(n_params_vec)