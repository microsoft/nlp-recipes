# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import torch
import torch.nn as nn


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


def parallelize_model(model, num_devices):
    """Implements model data parallelism on multiple GPUs.
    Args:
        model (PyTorch Module): A PyTorch model.
        num_devices (int): Number of GPUs to be used.    
    Returns:
        [DataParallel, Module]: A PyTorch DataParallel module wrapper
                                or a PyTorch Module (if multiple CUDA
                                devices are not available).
    """

    num_cuda_devices = torch.cuda.device_count()

    if num_cuda_devices < 2:
        print(
            "Warning: Only 1 CUDA device is available. Data parallelism is not possible."
        )
        return model

    if num_devices is None:
        num_devices = num_cuda_devices
    else:
        if num_devices < 2:
            return model
        if num_devices > num_cuda_devices:
            num_devices = num_cuda_devices
            print(
                "Warning: Only {} devices are available. Setting the number of devices to {}".format(
                    num_cuda_devices, num_cuda_devices
                )
            )

    if not isinstance(model, nn.DataParallel):
        return nn.DataParallel(model, device_ids=list(range(num_devices)))
    else:
        return model

