@triton.jit
def layernorm_kernel(
    x_ptr, y_ptr, weight_ptr, bias_ptr,
    N, D,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    """LayerNorm kernel - each program handles one sample."""
    # Program ID = which sample (row) we're processing
    row = tl.program_id(0)
    
    # Base pointers for this row
    x_row_ptr = x_ptr + row * D
    y_row_ptr = y_ptr + row * D
    
    # First pass: compute mean and variance
    # We'll accumulate in chunks of BLOCK_SIZE
    mean = 0.0
    var = 0.0
    
    for start_idx in range(0, D, BLOCK_SIZE):
        # Create offset mask
        offs = start_idx + tl.arange(0, BLOCK_SIZE)
        mask = offs < D
        
        # Load chunk of input
        x = tl.load(x_row_ptr + offs, mask=mask, other=0.0)
        
        # Accumulate sum for mean
        mean += tl.sum(x, axis=0)
    
    # Compute mean
    mean = mean / D
    
    # Second pass: compute variance
    for start_idx in range(0, D, BLOCK_SIZE):
        offs = start_idx + tl.arange(0, BLOCK_SIZE)
        mask = offs < D
        
        # Load chunk
        x = tl.load(x_row_ptr + offs, mask=mask, other=0.0)
        
        # Accumulate variance
        diff = x - mean
        var += tl.sum(diff * diff, axis=0)
    
    # Compute variance and standard deviation
    var = var / D
    std = tl.sqrt(var + eps)
    
    # Third pass: normalize and write output
    for start_idx in range(0, D, BLOCK_SIZE):
        offs = start_idx + tl.arange(0, BLOCK_SIZE)
        mask = offs < D
        
        # Load input, weight, and bias
        x = tl.load(x_row_ptr + offs, mask=mask, other=0.0)
        w = tl.load(weight_ptr + offs, mask=mask, other=1.0)
        b = tl.load(bias_ptr + offs, mask=mask, other=0.0)
        
        # Normalize
        x_norm = (x - mean) / std
        
        # Scale and shift
        y = w * x_norm + b
        
        # Store output
        tl.store(y_row_ptr + offs, y, mask=mask)
