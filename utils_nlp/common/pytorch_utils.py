# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Common PyTorch utilities that facilitate building PyTorch models."""

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler


def get_device(num_gpus=None, gpu_ids=None, local_rank=-1):
    if gpu_ids is not None:
        num_gpus = len(gpu_ids)
    if local_rank == -1:
        num_gpus = (
            min(num_gpus, torch.cuda.device_count())
            if num_gpus is not None
            else torch.cuda.device_count()
        )
        device = torch.device(
            "cuda" if torch.cuda.is_available() and num_gpus > 0 else "cpu"
        )
    else:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        num_gpus = 1
    return device, num_gpus


def move_model_to_device(model, device):
    if not isinstance(device, torch.device):
        raise ValueError("device must be of type torch.device.")

    # unwrap model
    # if isinstance(model, torch.nn.DataParallel):
    model = (
        model.module if hasattr(model, "module") else model
    )  # Take care of distributed/parallel training

    # move to device
    return model.to(device)


def parallelize_model(model, device, num_gpus=None, gpu_ids=None, local_rank=-1):
    """Moves a model to the specified device (cpu or gpu/s)
       and implements data parallelism when multiple gpus are specified.
    Args:
        model (Module): A PyTorch model.
        device (torch.device): A PyTorch device.
        num_gpus (int): The number of GPUs to be used.
            If set to None, all available GPUs will be used.
            Defaults to None.
        gpu_ids (list): List of GPU IDs to be used.
            If None, the first num_gpus GPUs will be used.
            If not None, overrides num_gpus. if gpu_ids is an empty list
            or there is no valid gpu devices are specified,
            and device is "cuda", model will not be moved or parallelized.
            Defaults to None.
        local_rank (int): Local GPU ID within a node. Used in distributed environments.
            If not -1, num_gpus and gpu_ids are ignored.
            Defaults to -1.
    Returns:
        Module, DataParallel, DistributedDataParallel: A PyTorch Module or
            a DataParallel/DistributedDataParallel wrapper,
            when one or multiple gpus are used.
    """
    if not isinstance(device, torch.device):
        raise ValueError("device must be of type torch.device.")

    model_module = (
        model.module if hasattr(model, "module") else model
    )  # Take care of distributed/parallel training

    if local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model_module,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
        )
    else:
        if device.type == "cuda":
            if num_gpus is not None:
                if num_gpus < 1:
                    raise ValueError("num_gpus must be at least 1 or None")
            num_cuda_devices = torch.cuda.device_count()
            if num_cuda_devices < 1:
                raise Exception("CUDA devices are not available.")
            if gpu_ids is None:
                num_gpus = (
                    num_cuda_devices
                    if num_gpus is None
                    else min(num_gpus, num_cuda_devices)
                )
                gpu_ids = list(range(num_gpus))
            else:
                gpu_ids = list(set(list(range(num_cuda_devices))).intersection(gpu_ids))
            if len(gpu_ids) > 0:
                model = torch.nn.DataParallel(model_module, device_ids=gpu_ids)
    return model


def dataloader_from_dataset(
    ds, batch_size=32, num_gpus=None, shuffle=False, distributed=False
):
    """Creates a PyTorch DataLoader given a Dataset object.

    Args:
        ds (torch.utils.data.DataSet): A PyTorch dataset.
        batch_size (int, optional): Batch size.
            If more than 1 gpu is used, this would be the batch size per gpu.
            Defaults to 32.
        num_gpus (int, optional): The number of GPUs to be used. Defaults to None.
        shuffle (bool, optional): If True, a RandomSampler is used. Defaults to False.
        distributed (book, optional): If True, a DistributedSampler is used.
        Defaults to False.

    Returns:
        Module, DataParallel: A PyTorch Module or
            a DataParallel wrapper (when multiple gpus are used).
    """
    if num_gpus is None:
        num_gpus = torch.cuda.device_count()

    batch_size = batch_size * max(1, num_gpus)

    if distributed:
        sampler = DistributedSampler(ds)
    else:
        sampler = RandomSampler(ds) if shuffle else SequentialSampler(ds)

    return DataLoader(ds, sampler=sampler, batch_size=batch_size)


def compute_training_steps(
    dataloader, num_epochs=1, max_steps=-1, gradient_accumulation_steps=1
):
    """Computes the max training steps given a dataloader.

    Args:
        dataloader (Dataloader): A PyTorch DataLoader.
        num_epochs (int, optional): Number of training epochs. Defaults to 1.
        max_steps (int, optional): Total number of training steps.
            If set to a positive value, it overrides num_epochs.
            Otherwise, it's determined by the dataset length,
            gradient_accumulation_steps, and num_epochs.
            Defaults to -1.
        gradient_accumulation_steps (int, optional): Number of steps to accumulate
            before performing a backward/update pass.
            Default to 1.

    Returns:
        int: The max number of steps to be used in a training loop.
    """
    try:
        dataset_length = len(dataloader)
    except Exception:
        dataset_length = -1
    if max_steps <= 0:
        if dataset_length != -1 and num_epochs > 0:
            max_steps = dataset_length // gradient_accumulation_steps * num_epochs
    if max_steps <= 0:
        raise Exception("Max steps cannot be determined.")
    return max_steps


def get_amp(fp16):
    """This function ensures that fp16 execution of torch.einsum is enabled
        if fp16 is set. Otherwise, it'll default to "promote" mode,
        where the operations are in fp32.
        Note that setting `fp16_opt_level="O2"` will remove the need for this code.
    """
    # Before we do anything with models, we want to
    if fp16:
        try:
            from apex import amp

            amp.register_half_function(torch, "einsum")
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex"
            )
    else:
        amp = None
    return amp
