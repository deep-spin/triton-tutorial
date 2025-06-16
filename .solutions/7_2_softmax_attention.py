@triton.jit
def attention_kernel_v2(
    Q_ptr, K_ptr, V_ptr, Out_ptr,
    B, H, N, d,
    scale,
    stride_qb, stride_qh, stride_qn, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_on, stride_od,
    causal: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    Flash Attention style - each program handles BLOCK_M queries.
    Uses online softmax to never materialize full attention.
    """
    # Program ID
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    # Decode batch and head
    batch_idx = pid_bh // H
    head_idx = pid_bh % H
    
    # Query block this program handles
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    
    # Initialize pointers
    Q_block_ptr = Q_ptr + batch_idx * stride_qb + head_idx * stride_qh
    K_block_ptr = K_ptr + batch_idx * stride_kb + head_idx * stride_kh  
    V_block_ptr = V_ptr + batch_idx * stride_vb + head_idx * stride_vh
    O_block_ptr = Out_ptr + batch_idx * stride_ob + head_idx * stride_oh
    
    # Load queries [BLOCK_M, d]
    mask_m = offs_m < N
    q_ptrs = Q_block_ptr + offs_m[:, None] * stride_qn + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=mask_m[:, None] & (offs_d[None, :] < d), other=0.0)
    
    # Initialize online softmax stats and output
    m_i = tl.full([BLOCK_M], value=-float('inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    
    # Loop over key/value blocks
    for start_n in range(0, N, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        
        # Load keys and values
        k_ptrs = K_block_ptr + offs_n[None, :] * stride_kn + offs_d[:, None] * stride_kd
        v_ptrs = V_block_ptr + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
        
        mask_n = offs_n < N
        k = tl.load(k_ptrs, mask=(offs_d[:, None] < d) & mask_n[None, :], other=0.0)
        
        # Compute scores [BLOCK_M, BLOCK_N]
        scores = tl.dot(q, k, out_dtype=tl.float32) * scale
        
        # Causal mask
        if causal:
            mask_causal = offs_m[:, None] >= offs_n[None, :]
            scores = tl.where(mask_causal & mask_n[None, :], scores, -float('inf'))
        else:
            scores = tl.where(mask_n[None, :], scores, -float('inf'))
        
        # Online softmax update
        m_ij = tl.max(scores, axis=1)  # [BLOCK_M]
        m_i_new = tl.maximum(m_i, m_ij)
        
        # Compute exponentials
        exp_scores = tl.exp(scores - m_i_new[:, None])
        l_ij = tl.sum(exp_scores, axis=1)
        
        # Update statistics
        alpha = tl.exp(m_i - m_i_new)
        l_i_new = alpha * l_i + l_ij
        
        # Load values
        v = tl.load(v_ptrs, mask=mask_n[:, None] & (offs_d[None, :] < d), other=0.0)
        
        # Update accumulator
        acc = acc * alpha[:, None] + tl.dot(exp_scores, v, out_dtype=tl.float32)
        
        # Update m_i and l_i for next iteration
        m_i = m_i_new
        l_i = l_i_new
    
    # Final output
    acc = acc / l_i[:, None]
    
    # Store
    o_ptrs = O_block_ptr + offs_m[:, None] * stride_on + offs_d[None, :] * stride_od
    tl.store(o_ptrs, acc, mask=mask_m[:, None] & (offs_d[None, :] < d))