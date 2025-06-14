# Triton Tutorial

This repository contains a series of puzzle-style Jupyter notebooks that guide you through implementing common GPU kernels first in raw PyTorch, then with `torch.compile`, and finally with Triton. 

## Links

- Triton docs: https://triton-lang.org/main/python-api/triton.html
- Triton tutorials: https://triton-lang.org/main/getting-started/tutorials/index.html


## Usage

First, install dependencies:

```bash
pip install -r requirements.txt
```

Next, open the notebooks in Jupyter, step through each cell, fill in the Triton puzzle, then benchmark and visualize.


## Puzzles Covered

1. Vector Addition
2. Fused Softmax
3. Matrix Multiplication
4. Layer Normalization
5. Fused Cross-Entropy Loss
6. Fused Softmax Attention
7. Fused Sparsemax / Entmax Attention

---

Happy puzzling! Contributions and improvements are welcome.
