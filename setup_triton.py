import sys, os

def setup_triton(use_interpreter=True, print_autotuning=False):
    # Add the local triton to your path (MacOS and local installation)
    triton_path = os.path.abspath("triton-cpu/python")
    if os.path.isdir(triton_path) and triton_path not in sys.path:
        sys.path.append(triton_path)

    # TRITON_INTERPRET=1 uses a python interpreter instead of running on the GPU. 
    # This menas that uou can insert Python breakpoints to debug your kernel code! 
    if use_interpreter:
        os.environ["TRITON_INTERPRET"] = "1"

    # Enable printing of autotuning info: best configuration and benchmark time
    if print_autotuning:
        os.environ["TRITON_PRINT_AUTOTUNING"] = "1"
