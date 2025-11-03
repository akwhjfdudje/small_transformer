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

A model requires a dataset of text to use:

To do so, run this in this directory:
```
python generators/*.py 
```

To train and save a model:

```
python train.py
```

The `evaluate.py` script takes transformer and vocabulary checkpoints, and will give a small prompt to input text:

```
python evaluate.py
```
