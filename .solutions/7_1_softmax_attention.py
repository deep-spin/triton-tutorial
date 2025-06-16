@triton.jit
def attention_kernel_v1(
    Q_ptr, K_ptr, V_ptr, Out_ptr,
    B, H, N, d,
    scale,
    stride_qb, stride_qh, stride_qn, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_on, stride_od,
    causal: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Simple attention kernel - one program per query."""
    # Get program ID and decode indices
    pid = tl.program_id(0)
    
    # Decode batch, head, and query indices
    query_idx = pid % N
    head_idx = (pid // N) % H
    batch_idx = pid // (N * H)
    
    # Pointers to Q[batch, head, query, :]
    Q_ptr += batch_idx * stride_qb + head_idx * stride_qh + query_idx * stride_qn
    
    # Pointers to K[batch, head, :, :] and V[batch, head, :, :]
    K_ptr += batch_idx * stride_kb + head_idx * stride_kh
    V_ptr += batch_idx * stride_vb + head_idx * stride_vh
    
    # Output pointer
    Out_ptr += batch_idx * stride_ob + head_idx * stride_oh + query_idx * stride_on
    
    # Load query vector
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < d
    q = tl.load(Q_ptr + offs_d * stride_qd, mask=mask_d, other=0.0)
    
    # Initialize max and sum for online softmax
    max_score = -float('inf')
    sum_exp = 0.0
    
    # Initialize output accumulator
    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
    
    # First pass: compute softmax statistics
    for start_n in range(0, N, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N
        
        # Causal mask
        if causal:
            mask_n = mask_n & (offs_n <= query_idx)
        
        # Load keys [BLOCK_N, d]
        k_ptrs = K_ptr + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
        k = tl.load(k_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0)
        
        # Compute scores for this block
        scores = tl.sum(q[None, :] * k, axis=1) * scale  # [BLOCK_N]
        
        # Apply mask
        scores = tl.where(mask_n, scores, -float('inf'))
        
        # Update max
        block_max = tl.max(scores, axis=0)
        max_score = tl.maximum(max_score, block_max)
    
    # Second pass: compute attention weights and output
    for start_n in range(0, N, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N
        
        # Causal mask
        if causal:
            mask_n = mask_n & (offs_n <= query_idx)
        
        # Recompute keys and scores
        k_ptrs = K_ptr + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
        k = tl.load(k_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0)
        scores = tl.sum(q[None, :] * k, axis=1) * scale
        scores = tl.where(mask_n, scores, -float('inf'))
        
        # Compute exp(scores - max)
        exp_scores = tl.exp(scores - max_score)
        exp_scores = tl.where(mask_n, exp_scores, 0.0)
        sum_exp += tl.sum(exp_scores)
        
    # Third pass: compute weighted sum
    for start_n in range(0, N, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N
        
        # Causal mask
        if causal:
            mask_n = mask_n & (offs_n <= query_idx)
        
        # Recompute keys and scores (yes, again!)
        k_ptrs = K_ptr + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
        k = tl.load(k_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0)
        scores = tl.sum(q[None, :] * k, axis=1) * scale
        scores = tl.where(mask_n, scores, -float('inf'))
        
        # Compute attention weights
        weights = tl.exp(scores - max_score) / sum_exp
        weights = tl.where(mask_n, weights, 0.0)
        
        # Load values
        v_ptrs = V_ptr + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0)
        
        # Accumulate weighted values
        acc += tl.sum(weights[:, None] * v, axis=0)
    
    # Store output
    tl.store(Out_ptr + offs_d * stride_od, acc, mask=mask_d)