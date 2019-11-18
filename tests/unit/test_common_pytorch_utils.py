# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.


import pytest
import torch
import torch.nn as nn
from torch.nn.parallel.data_parallel import DataParallel
from torch.nn.modules.container import Sequential

from utils_nlp.common.pytorch_utils import get_device, move_to_device


@pytest.fixture
def model():
    return nn.Sequential(nn.Linear(24, 8), nn.ReLU(), nn.Linear(8, 2), nn.Sigmoid())


def test_get_device_cpu():
    device, gpus = get_device(num_gpus=0)
    assert isinstance(device, torch.device)
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


@pytest.mark.gpu
def test_get_device_all_gpus():
    device, gpus = get_device()
    assert isinstance(device, torch.device)
    assert device.type == "cuda"
    assert gpus == torch.cuda.device_count()


@pytest.mark.gpu
def test_get_device_local_rank():
    device, gpus = get_device(local_rank=1)
    assert isinstance(device, torch.device)
    assert device.type == "cuda"
    assert device.index == 1
    assert gpus == 1


def test_move_to_device_cpu(model):
    # test when device.type="cpu"
    model_cpu = move_to_device(model, torch.device("cpu"))
    assert isinstance(model_cpu, nn.modules.container.Sequential)


def test_move_to_device_cpu_parallelized(model):
    # test when input model is parallelized
    model_parallelized = nn.DataParallel(model)
    model_parallelized_output = move_to_device(model_parallelized, torch.device("cpu"))
    assert isinstance(model_parallelized_output, nn.modules.container.Sequential)


def test_move_to_device_exception_not_torch_device(model):
    # test when device is not torch.device
    with pytest.raises(ValueError):
        move_to_device(model, "abc")


def test_move_to_device_exception_wrong_type(model):
    # test when device.type is not "cuda" or "cpu"
    with pytest.raises(Exception):
        move_to_device(model, torch.device("opengl"))


@pytest.mark.skipif(
    torch.cuda.is_available(), reason="Skip if we are executing the cpu tests on a gpu machine"
)
def test_move_to_device_exception_gpu_model_on_cpu_machine(model):
    # test when the model is moved to a gpu but it is a cpu machine
    with pytest.raises(Exception):
        move_to_device(model, torch.device("cuda"))


@pytest.mark.gpu
def test_move_to_device_exception_cuda_zero_gpus(model):
    # test when device.type is cuda, but num_gpus is 0
    with pytest.raises(ValueError):
        move_to_device(model, torch.device("cuda"), num_gpus=0)


@pytest.mark.gpu
def test_move_to_device_gpu(model):
    # test when device.type="cuda"
    model_cuda = move_to_device(model, torch.device("cuda"))
    num_cuda_devices = torch.cuda.device_count()

    if num_cuda_devices > 1:
        assert isinstance(model_cuda, DataParallel)
    else:
        assert isinstance(model_cuda, Sequential)

    model_cuda_1_gpu = move_to_device(model, torch.device("cuda"), num_gpus=1)
    assert isinstance(model_cuda_1_gpu, Sequential)

    model_cuda_1_more_gpu = move_to_device(
        model, torch.device("cuda"), num_gpus=num_cuda_devices + 1
    )
    if num_cuda_devices > 1:
        assert isinstance(model_cuda_1_more_gpu, DataParallel)
    else:
        assert isinstance(model_cuda_1_more_gpu, Sequential)

    model_cuda_same_gpu = move_to_device(model, torch.device("cuda"), num_gpus=num_cuda_devices)
    if num_cuda_devices > 1:
        assert isinstance(model_cuda_same_gpu, DataParallel)
    else:
        assert isinstance(model_cuda_same_gpu, Sequential)

