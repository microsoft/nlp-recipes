### [Common](common)

This submodule contains high-level utilities for defining constants used in most algorithms as well as helper functions for managing aspects of different frameworks like pytorch.  

It contains the following scripts:  

1. [pytorch_utils.py](./pytorch_utils.py) - This contains utilities to interact with PyTorch like getting a device architecture (cpu or gpu), moves a model to a specific device which in turn handles parallelism when multiple gpus are present.  

1. [timer.py](./timer.py) - This contains timer utilities. It comes useful when benchmarking running times of executions. It exposes functionalities like `starting`, `stopping` and finding time `intervals`.