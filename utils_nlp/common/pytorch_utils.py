# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Common PyTorch utilities that facilitate building Pytorch models."""

import torch
import torch.nn as nn
import warnings


def get_device(
    num_gpus=None,
    local_rank=-1,
    #    backend="nccl",
    #    rank=0,
    #    world_size=1,
    #    init_method="file:///distributed",
):
    if local_rank == -1:
        num_gpus = (
            min(num_gpus, torch.cuda.device_count())
            if num_gpus is not None
            else torch.cuda.device_count()
        )
        device = torch.device("cuda" if torch.cuda.is_available() and num_gpus > 0 else "cpu")
    else:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        # torch.distributed.init_process_group(backend="nccl")
        # torch.distributed.init_process_group(backend=backend, rank=rank, world_size=world_size, init_method=init_method)
        num_gpus = 1
    return device, num_gpus


def move_to_device(model, device, num_gpus=None):
    """Moves a model to the specified device (cpu or gpu/s)
       and implements data parallelism when multiple gpus are specified.

    Args:
        model (Module): A PyTorch model
        device (torch.device): A PyTorch device
        num_gpus (int): The number of GPUs to be used. Defaults to None,
            all gpus are used.

    Returns:
        Module, DataParallel: A PyTorch Module or
            a DataParallel wrapper (when multiple gpus are used).
    """
    if isinstance(model, nn.DataParallel):
        model = model.module

    if not isinstance(device, torch.device):
        raise ValueError("device must be of type torch.device.")

    if device.type == "cuda":
        model.to(device)  # inplace
        if num_gpus == 0:
            raise ValueError("num_gpus must be non-zero when device.type is 'cuda'")
        elif num_gpus == 1:
            return model
        else:
            # parallelize
            num_cuda_devices = torch.cuda.device_count()
            if num_cuda_devices < 1:
                raise Exception("CUDA devices are not available.")
            elif num_cuda_devices < 2:
                print("Warning: Only 1 CUDA device is available. Data parallelism is not possible.")
                return model
            else:
                if num_gpus is None:
                    # use all available devices
                    return nn.DataParallel(model, device_ids=None)
                elif num_gpus > num_cuda_devices:
                    print(
                        "Warning: Only {0} devices are available. "
                        "Setting the number of gpus to {0}".format(num_cuda_devices)
                    )
                    return nn.DataParallel(model, device_ids=None)
                else:
                    return nn.DataParallel(model, device_ids=list(range(num_gpus)))
    elif device.type == "cpu":
        if num_gpus != 0 and num_gpus is not None:
            warnings.warn("Device type is 'cpu'. num_gpus is ignored.")
        return model.to(device)

    else:
        raise Exception(
            "Device type '{}' not supported. Currently, only cpu "
            "and cuda devices are supported.".format(device.type)
        )
