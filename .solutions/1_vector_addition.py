@triton.jit
def add_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for element-wise vector addition.
    Each program instance processes BLOCK_SIZE elements.
    """
    # Identify which program we are
    pid = tl.program_id(axis=0)
    
    # Calculate the starting index for this program
    block_start = pid * BLOCK_SIZE
    
    # Create a range of offsets for this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create a mask to guard against out-of-bounds accesses
    mask = offsets < n_elements
    
    # Load x and y from HBM to SRAM
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    
    # Perform the addition on SRAM
    output = x + y
    
    # Write the result back to HBM
    tl.store(output_ptr + offsets, output, mask=mask)
