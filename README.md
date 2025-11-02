# Small Language Model

This project contains an inference-based language model, using some of my own handwritten CUDA kernels.

## Building

To build the PyTorch extension:

```
python setup.py build_ext --inplace
cp <path/to/cuda_kernels/shared/lib> .
```

This will create the extension in the current directory

## Running

To train the model:

```
python train.py
```
