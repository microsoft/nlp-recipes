# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import torch


def get_device(device="gpu"):
    """Gets a PyTorch device.
    Args:
        device (str, optional): Device string: "cpu" or "gpu". Defaults to "gpu".
    Returns:
        torch.device: A PyTorch device: cpu or gpu.
    """
    if device == "gpu":
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        raise Exception("CUDA device not available")
    elif device == "cpu":
        return torch.device("cpu")
    else:
        raise Exception("Only 'cpu' and 'gpu' devices are supported.")
