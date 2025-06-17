# Triton Tutorial

This repository contains a series of puzzle-style Jupyter notebooks that guide you through implementing common GPU kernels first in raw PyTorch, then with `torch.compile`, and finally with [Triton](https://triton-lang.org/main/python-api/triton.html). 


<img src="figs/sardine-tecnico.png" width="830" />



## Installation

First, install dependencies:

```bash
pip install -r requirements.txt
```

### Linux/Windows

Install Triton directly via pip:
```
pip install triton==3.3.1
```

### MacOS

You need to build from source (it takes some time -- ~15min on my Mac):

```bash
git clone https://github.com/triton-lang/triton-cpu.git
cd triton-cpu
git submodule update --init --recursive
cd python
pip install -r requirements.txt
pip install -e .
```



## Puzzles Covered

Now, just open the notebooks in Jupyter, step through each cell and solve the puzzle!

1. Vector Addition
2. Fused Softmax
3. Fused Entmax
4. Matrix Multiplication
5. Layer Normalization
6. Cross-Entropy Loss
7. Softmax Attention - Forward Pass
8. Sparsemax Attention - Forward Pass (*BONUS*)

Happy puzzling! 

---

## Further References

- [Triton Docs](https://triton-lang.org/main/python-api/triton.html)
- [Triton Tutorial](https://triton-lang.org/main/getting-started/tutorials/index.html)
- [GPU Mode](https://www.youtube.com/@GPUMODE) (lecture 14)
- [Christian Mills' Notes](https://christianjmills.com/posts/cuda-mode-notes/lecture-014/)

Kernels:

- [LightLLM](https://github.com/ModelTC/lightllm/tree/main/lightllm/common/basemodel/triton_kernel)
- [Unsloth](https://github.com/unslothai/unsloth/tree/main/unsloth/kernels) 
- [Liger](https://github.com/linkedin/Liger-Kernel/tree/main/src/liger_kernel/ops) 


