@triton.jit
def alpha_entmax(x, tau, alpha):
    x = (alpha - 1) * x - tau
    # Here we have to mask out negative values
    # because we are using log2, which is not defined for negative values.
    x = tl.where(x > 0, tl.exp2(1 / (alpha - 1) * tl.log2(x)), 0.0)
    return x


@triton.jit
def _ent_bisect(
    x_ptr, 
    y_ptr, 
    alpha, 
    n_iter, 
    N: tl.constexpr, 
    TILE: tl.constexpr
):
    # get row that this thread block will be responsible for
    curr_row = tl.program_id(0)

    # move pointers to the start of the input and output tensors
    x_ptr += curr_row * N
    y_ptr += curr_row * N
    
    # same as torch.arange
    offsets = tl.arange(0, TILE)

    # placeholder for max value
    max_val = -1.0e3

    for idx in range(0, N, TILE):
        # compute pointers for the current tile
        x_ptrs = (x_ptr + idx) + offsets

        # load TILE elements of X
        x = tl.load(x_ptrs)

        # update max value
        max_val = tl.maximum(max_val, tl.max(x))

    max_val *= (alpha - 1.0)

    # initialize tau bounds
    tau_lower = max_val - 1.0
    tau_upper = max_val - tl.exp2((1-alpha) * tl.log2(1.0*N))
    tau = (tau_lower + tau_upper) / 2.0
    
    # bisection
    for _ in range(n_iter):
        f_tau = -1.0

        for idx in range(0, N, TILE):
            # compute pointers for the current tile
            x_ptrs = (x_ptr + idx) + offsets

            # load TILE elements of X
            x = tl.load(x_ptrs)

            # accumulate f(tau)
            f_tau += tl.sum(alpha_entmax(x, tau, alpha))

        # update tau bounds
        if f_tau > 0:
            tau_lower = tau
        else:
            tau_upper = tau
        tau = (tau_lower + tau_upper) / 2.0

    # apply tau
    for idx in range(0, N, TILE):
            # compute pointers for the current tile
            x_ptrs = (x_ptr + idx) + offsets
            y_ptrs = (y_ptr + idx) + offsets

            # load TILE elements of X
            x = tl.load(x_ptrs)

            # compute entmax for this TILE
            y = alpha_entmax(x, tau, alpha)

            # store results
            tl.store(y_ptrs, y)
