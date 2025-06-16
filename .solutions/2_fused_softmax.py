@triton.jit
def softmax_kernel(
    input_ptr,
    output_ptr,
    n_cols,
    input_stride,
    output_stride,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused softmax kernel. Each program computes one row.
    """
    # Get the row index for this program
    row_idx = tl.program_id(0)
    
    # Calculate the starting pointer for this row
    input_row_start = input_ptr + row_idx * input_stride
    output_row_start = output_ptr + row_idx * output_stride
    
    # Step 1: Find maximum value in the row
    # We process in chunks of BLOCK_SIZE
    row_max = float('-inf')
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        # Create column indices for this block
        cols = col_offset + tl.arange(0, BLOCK_SIZE)
        
        # Mask to handle the last block if n_cols % BLOCK_SIZE != 0
        mask = cols < n_cols
        
        # Load values from HBM to SRAM
        vals = tl.load(input_row_start + cols, mask=mask, other=float('-inf'))
        
        # Update maximum
        row_max = tl.maximum(row_max, tl.max(vals, axis=0))
    
    # Step 2: Compute exp(x - max) and sum
    # We need another pass through the data
    exp_sum = 0.0
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        cols = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        
        # Load values again
        vals = tl.load(input_row_start + cols, mask=mask, other=float('-inf'))
        
        # Compute exp(x - max) for numerical stability
        exp_vals = tl.exp(vals - row_max)
        
        # Mask out invalid values
        exp_vals = tl.where(mask, exp_vals, 0.0)
        
        # Add to sum
        exp_sum += tl.sum(exp_vals, axis=0)
    
    # Step 3: Normalize and store
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        cols = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        
        # Load values one more time
        vals = tl.load(input_row_start + cols, mask=mask, other=float('-inf'))
        
        # Compute final softmax values
        exp_vals = tl.exp(vals - row_max)
        softmax_vals = exp_vals / exp_sum
        
        # Store results back to HBM
        tl.store(output_row_start + cols, softmax_vals, mask=mask)
