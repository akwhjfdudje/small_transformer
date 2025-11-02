# Small Language Model

This project contains an inference-based language model, using some of my own handwritten CUDA kernels.

## Building

To build the PyTorch extension:

```
cmake -S . -B build \
-DCMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')" \
-DPython3_ROOT_DIR="<path/to/python>"
-DTORCH_PATH="<path/to/pytorch>"
```
