# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""PyTorch utils tests."""

import pytest
import torch
import torch.nn as nn
from torch.nn.parallel.data_parallel import DataParallel

from utils_nlp.common.pytorch_utils import (
    get_device,
    move_model_to_device,
    parallelize_model,
)


@pytest.fixture
def model():
    return nn.Sequential(nn.Linear(24, 8), nn.ReLU(), nn.Linear(8, 2), nn.Sigmoid())


def test_get_device_cpu():
    device, gpus = get_device(num_gpus=0)
    assert isinstance(device, torch.device)
    assert device.type == "cpu"
    assert gpus == 0

    device, gpus = get_device(gpu_ids=[])
    assert device.type == "cpu"
    assert gpus == 0


@pytest.mark.gpu
def test_machine_is_gpu_machine():
    assert torch.cuda.is_available() is True


@pytest.mark.gpu
def test_get_device_gpu():
    device, gpus = get_device(num_gpus=1)
    assert isinstance(device, torch.device)
    assert device.type == "cuda"
    assert gpus == 1

    device, gpus = get_device(gpu_ids=[0])
    assert device.type == "cuda"
    assert gpus == 1


@pytest.mark.gpu
def test_get_device_all_gpus():
    device, gpus = get_device()
    assert isinstance(device, torch.device)
    assert device.type == "cuda"
    assert gpus == torch.cuda.device_count()


@pytest.mark.gpu
def test_get_device_local_rank():
    device, gpus = get_device(local_rank=0)
    assert isinstance(device, torch.device)
    assert device.type == "cuda"
    assert device.index == 0
    assert gpus == 1


def test_get_device_local_rank_cpu():
    device, gpus = get_device(local_rank=-1, num_gpus=0)
    assert isinstance(device, torch.device)
    assert device.type == "cpu"
    assert gpus == 0


def test_move_to_device_cpu(model):
    # test when device.type="cpu"
    model_cpu = move_model_to_device(model, torch.device("cpu"))
    assert isinstance(model_cpu, nn.modules.container.Sequential)
    assert next(model_cpu.parameters()).is_cuda is False


def test_move_to_device_cpu_parallelized(model):
    # test when input model is parallelized
    model_parallelized = nn.DataParallel(model)
    model_parallelized_output = move_model_to_device(
        model_parallelized, torch.device("cpu")
    )
    assert isinstance(model_parallelized_output, nn.modules.container.Sequential)
    assert next(model_parallelized.module.parameters()).is_cuda is False


def test_move_to_device_exception_not_torch_device(model):
    # test when device is not torch.device
    with pytest.raises(ValueError):
        move_model_to_device(model, "abc")


def test_move_to_device_exception_wrong_type(model):
    # test when device.type is not "cuda" or "cpu"
    with pytest.raises(Exception):
        move_model_to_device(model, torch.device("opengl"))


@pytest.mark.skipif(
    torch.cuda.is_available(),
    reason="Skip if we are executing the cpu tests on a gpu machine",
)
def test_move_to_device_exception_gpu_model_on_cpu_machine(model):
    # test when the model is moved to a gpu but it is a cpu machine
    with pytest.raises(Exception):
        move_model_to_device(model, torch.device("cuda"))


@pytest.mark.gpu
def test_parallelize_model_exception_cuda_zero_gpus(model):
    # test when device.type is cuda, but num_gpus is 0
    with pytest.raises(ValueError):
        model = move_model_to_device(model, torch.device("cuda"))
        parallelize_model(model, torch.device("cuda"), num_gpus=0)


@pytest.mark.gpu
def test_parallelize_model(model):
    # test when device.type="cuda" and move model to all devices
    model_cuda = move_model_to_device(model, torch.device("cuda"))
    model_cuda = parallelize_model(model_cuda, torch.device("cuda"))
    num_cuda_devices = torch.cuda.device_count()
    assert isinstance(model_cuda, DataParallel)

    # test moving model to only one gpu
    model_cuda_1_gpu = move_model_to_device(model, torch.device("cuda"))
    assert next(model_cuda_1_gpu.parameters()).is_cuda is True
    model_cuda_1_gpu = parallelize_model(
        model_cuda_1_gpu, torch.device("cuda"), num_gpus=1
    )
    assert next(model_cuda_1_gpu.parameters()).is_cuda is True

    # test parallelize_model can limit the number of devices
    model_cuda_1_more_gpu = move_model_to_device(model, torch.device("cuda"))
    model_cuda_1_more_gpu = parallelize_model(
        model_cuda_1_more_gpu, torch.device("cuda"), num_gpus=num_cuda_devices + 1
    )
    assert next(model_cuda_1_more_gpu.module.parameters()).is_cuda is True

    # test parallelize_model on the same number of devices
    model_cuda_same_gpu = move_model_to_device(model, torch.device("cuda"))
    model_cuda_same_gpu = parallelize_model(
        model_cuda_same_gpu, torch.device("cuda"), num_gpus=num_cuda_devices
    )
    assert next(model_cuda_same_gpu.module.parameters()).is_cuda is True

    # test parallelize_model with gpu id
    model_base = move_model_to_device(model, torch.device("cuda"))
    # when gpu id is [], gpu id [0] is used
    model_cuda_0_gpu = parallelize_model(model_base, torch.device("cuda"), gpu_ids=[])
    # device has priority ??
    assert next(model_cuda_1_gpu.parameters()).device == torch.device("cuda:0")
    assert next(model_cuda_0_gpu.parameters()).is_cuda is True

    # test parallelize_model with gpu id is [0]
    model_base = move_model_to_device(model, torch.device("cuda"))
    model_cuda_1_gpu = parallelize_model(model_base, torch.device("cuda"), gpu_ids=[0])
    assert next(model_cuda_1_gpu.parameters()).is_cuda is True

    # test parallelize_model with gpu id is [0:num_device]
    model_base = move_model_to_device(model, torch.device("cuda"))
    model_cuda_same_gpu = parallelize_model(
        model_base, torch.device("cuda"), gpu_ids=list(range(num_cuda_devices))
    )
    if num_cuda_devices > 1:
        assert next(model_cuda_same_gpu.module.parameters()).is_cuda is True
    else:
        assert next(model_cuda_same_gpu.parameters()).is_cuda is True

    # test parallelize_model with gpu id is [1: num_devices+3]
    model_base = move_model_to_device(model, torch.device("cuda"))
    model_cuda_same_gpu = parallelize_model(
        model_base,
        torch.device("cuda"),
        gpu_ids=[x + 1 for x in list(range(num_cuda_devices + 2))],
    )
    if num_cuda_devices > 1:
        assert next(model_cuda_same_gpu.module.parameters()).is_cuda is True
    else:
        assert next(model_cuda_same_gpu.parameters()).is_cuda is True

    # when intersection is only 1
    model_base = move_model_to_device(model, torch.device("cuda"))
    gpu_ids = [x + num_cuda_devices - 1 for x in list(range(num_cuda_devices))]
    model_cuda_intersect_1_gpu = parallelize_model(
        model_base, torch.device("cuda"), gpu_ids=gpu_ids
    )
    assert next(model_cuda_intersect_1_gpu.parameters()).device == torch.device(
        "cuda:{}".format(num_cuda_devices - 1)
    )
    assert next(model_cuda_intersect_1_gpu.parameters()).is_cuda is True

    # when threre is no intersection, no change to the model 
    model_base = move_model_to_device(model, torch.device("cuda"))
    model_cuda_intersect_0_gpu = parallelize_model(
        model_base,
        torch.device("cuda"),
        gpu_ids=[x + num_cuda_devices for x in list(range(num_cuda_devices))],
    )
    assert (
        next(model_cuda_intersect_0_gpu.parameters()).device
        == next(model_base.parameters()).device
    )
    assert next(model_cuda_intersect_0_gpu.parameters()).is_cuda is True
    # test device is cpu original model on gpu
    model_base = move_model_to_device(model, torch.device("cuda"))
    model_cuda_cpu = parallelize_model(
        model_base,
        torch.device("cpu"),
        gpu_ids=[x + num_cuda_devices for x in list(range(num_cuda_devices))],
    )
    assert next(model_cuda_cpu.parameters()).is_cuda is True
    # test device is cpu and original model on cpu
    model_base = move_model_to_device(model, torch.device("cpu"))
    model_cuda_cpu = parallelize_model(
        model_base,
        torch.device("cpu"),
        gpu_ids=[x + num_cuda_devices for x in list(range(num_cuda_devices))],
    )
    assert next(model_cuda_cpu.parameters()).is_cuda is False
