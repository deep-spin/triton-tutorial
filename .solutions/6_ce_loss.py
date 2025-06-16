@triton.jit
def cross_entropy_forward_kernel(
    logits_ptr, labels_ptr, losses_ptr,
    N, C,
    BLOCK_SIZE: tl.constexpr
):
    """Forward kernel for cross entropy loss."""
    # Program ID = sample index
    row = tl.program_id(0)
    
    # Base pointers
    logits_row_ptr = logits_ptr + row * C
    
    # First pass: find max for numerical stability
    max_logit = -float('inf')
    for start_idx in range(0, C, BLOCK_SIZE):
        offs = start_idx + tl.arange(0, BLOCK_SIZE)
        mask = offs < C
        
        # Load chunk of logits
        chunk = tl.load(logits_row_ptr + offs, mask=mask, other=-float('inf'))
        
        # Update max
        max_logit = tl.max(tl.maximum(chunk, max_logit))
    
    # Second pass: compute log-sum-exp
    sum_exp = 0.0
    for start_idx in range(0, C, BLOCK_SIZE):
        offs = start_idx + tl.arange(0, BLOCK_SIZE)
        mask = offs < C
        
        # Load chunk and compute exp
        chunk = tl.load(logits_row_ptr + offs, mask=mask, other=-float('inf'))
        exp_chunk = tl.exp(chunk - max_logit)
        
        # Accumulate sum
        sum_exp += tl.sum(tl.where(mask, exp_chunk, 0.0))
    
    # Compute log-sum-exp
    log_sum_exp = max_logit + tl.log(sum_exp)
    
    # Get the label for this sample
    label = tl.load(labels_ptr + row)
    
    # Get the logit for the correct class
    correct_logit = tl.load(logits_row_ptr + label)
    
    # Compute loss
    loss = log_sum_exp - correct_logit
    
    # Store result
    tl.store(losses_ptr + row, loss)


@triton.jit
def cross_entropy_backward_kernel(
    grad_output_ptr, logits_ptr, labels_ptr, grad_input_ptr,
    N, C,
    BLOCK_SIZE: tl.constexpr
):
    """Backward kernel for cross entropy loss."""
    # Program ID = sample index
    row = tl.program_id(0)
    
    # Load gradient w.r.t. loss
    grad_out = tl.load(grad_output_ptr + row)
    
    # Base pointers
    logits_row_ptr = logits_ptr + row * C
    grad_row_ptr = grad_input_ptr + row * C
    
    # Load label
    label = tl.load(labels_ptr + row)
    
    # First pass: find max for numerical stability
    max_logit = -float('inf')
    for start_idx in range(0, C, BLOCK_SIZE):
        offs = start_idx + tl.arange(0, BLOCK_SIZE)
        mask = offs < C
        
        chunk = tl.load(logits_row_ptr + offs, mask=mask, other=-float('inf'))
        max_logit = tl.max(tl.maximum(chunk, max_logit))
    
    # Second pass: compute sum of exp
    sum_exp = 0.0
    for start_idx in range(0, C, BLOCK_SIZE):
        offs = start_idx + tl.arange(0, BLOCK_SIZE)
        mask = offs < C
        
        chunk = tl.load(logits_row_ptr + offs, mask=mask, other=-float('inf'))
        exp_chunk = tl.exp(chunk - max_logit)
        sum_exp += tl.sum(tl.where(mask, exp_chunk, 0.0))

    # Third pass: compute gradients
    for start_idx in range(0, C, BLOCK_SIZE):
        offs = start_idx + tl.arange(0, BLOCK_SIZE)
        mask = offs < C
        
        # Load logits and compute softmax
        logits = tl.load(logits_row_ptr + offs, mask=mask, other=-float('inf'))
        softmax = tl.exp(logits - max_logit) / sum_exp

        # Compute gradient: softmax - delta_{iy}
        delta = tl.where(offs == label, 1.0, 0.0)
        grad = softmax - delta
        
        # Scale by upstream gradient
        grad_new = grad * grad_out
        
        # Store
        tl.store(grad_row_ptr + offs, grad_new, mask=mask)
