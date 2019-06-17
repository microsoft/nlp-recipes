# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""PyTorch device utils."""

import torch
import torch.nn as nn


def get_device(device="gpu"):
    """Gets a PyTorch device.
    Args:
        device (str, optional): Device string: "cpu" or "gpu". Defaults to "gpu".
    Returns:
        torch.device: A PyTorch device (cpu or gpu).
    """
    if device == "gpu":
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        raise Exception("CUDA device not available")
    elif device == "cpu":
        return torch.device("cpu")
    else:
        raise Exception("Only 'cpu' and 'gpu' devices "
                        "are supported. device was {}.".format(device))


def move_to_device(model, device, num_gpus=1):
    """Moves a model to the specified device (cpu or gpu/s)
       and implements data parallelism when multiple gpus are specified.
    Args:
        model (Module): A PyTorch model
        device (torch.device): A PyTorch device
        num_gpus (int): The number of GPUs to be used. Defaults to 1.
    Returns:
        Module, DataParallel: A PyTorch Module or a DataParallel wrapper (when multiple gpus are used).
    """
    if isinstance(model, nn.DataParallel):
        model = model.module

    # cpu
    if num_gpus == 0:
        if device.type == "cpu":
            return model.to(device)
        else:
            raise Exception("Device type should be 'cpu' when num_gpus==0.")

    # gpu
    if device.type == "cuda":
        model.to(device)  # inplace
        if num_gpus == 1:
            return model
        else:
            # parallelize
            num_cuda_devices = torch.cuda.device_count()
            if num_cuda_devices < 1:
                raise Exception("CUDA devices are not available.")
            elif num_cuda_devices < 2:
                print(
                    "Warning: Only 1 CUDA device is available. Data parallelism is not possible."
                )
                return model
            else:
                if num_gpus is None:
                    # use all available devices
                    return nn.DataParallel(model, device_ids=None)
                elif num_gpus > num_cuda_devices:
                    print(
                        "Warning: Only {0} devices are available. Setting the number of gpus to {0}".format(
                            num_cuda_devices
                        )
                    )
                    return nn.DataParallel(model, device_ids=None)
                else:
                    return nn.DataParallel(
                        model, device_ids=list(range(num_gpus))
                    )
    else:
        raise Exception("Device type should be 'gpu' when num_gpus!=0.")
