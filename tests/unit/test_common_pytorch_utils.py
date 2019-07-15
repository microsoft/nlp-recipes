# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pytest
import torch
import torch.nn as nn
from utils_nlp.common.pytorch_utils import get_device, move_to_device


def test_get_device():

    if torch.cuda.is_available():
        device = get_device()
        assert isinstance(device, torch.device)
        assert device.type == "cuda"
    else:
        with pytest.raises(Exception):
            get_device()

        device = get_device("cpu")
        assert isinstance(device, torch.device)
        assert device.type == "cpu"

    with pytest.raises(ValueError):
        get_device("abc")


def test_move_to_device():
    model = nn.Sequential(
        nn.Linear(24, 8), nn.ReLU(), nn.Linear(8, 2), nn.Sigmoid()
    )

    # test when input model is parallelized
    model_parallelized = nn.DataParallel(model)
    model_parallelized_output = move_to_device(
        model_parallelized, torch.device("cpu")
    )
    assert isinstance(
        model_parallelized_output, nn.modules.container.Sequential
    )

    # test when device is not torch.device
    with pytest.raises(ValueError):
        move_to_device(model, "abc")

    if torch.cuda.is_available():
        # test when device.type="cuda"
        model_cuda = move_to_device(model, torch.device("cuda"))
        num_cuda_devices = torch.cuda.device_count()

        if num_cuda_devices > 1:
            assert isinstance(
                model_cuda, nn.parallel.data_parallel.DataParallel
            )
        else:
            assert isinstance(model_cuda, nn.modules.container.Sequential)

        model_cuda_1_gpu = move_to_device(
            model, torch.device("cuda"), num_gpus=1
        )
        assert isinstance(model_cuda_1_gpu, nn.modules.container.Sequential)

        model_cuda_1_more_gpu = move_to_device(
            model, torch.device("cuda"), num_gpus=num_cuda_devices + 1
        )
        if num_cuda_devices > 1:
            assert isinstance(
                model_cuda_1_more_gpu, nn.parallel.data_parallel.DataParallel
            )
        else:
            assert isinstance(
                model_cuda_1_more_gpu, nn.modules.container.Sequential
            )

        model_cuda_same_gpu = move_to_device(
            model, torch.device("cuda"), num_gpus=num_cuda_devices
        )
        if num_cuda_devices > 1:
            assert isinstance(
                model_cuda_same_gpu, nn.parallel.data_parallel.DataParallel
            )
        else:
            assert isinstance(
                model_cuda_same_gpu, nn.modules.container.Sequential
            )

        # test when device.type is cuda, but num_gpus is 0
        with pytest.raises(ValueError):
            move_to_device(model, torch.device("cuda"), num_gpus=0)
    else:
        with pytest.raises(Exception):
            move_to_device(model, torch.device("cuda"))

    # test when device.type="cpu"
    model_cpu = move_to_device(model, torch.device("cpu"))
    assert isinstance(model_cpu, nn.modules.container.Sequential)

    # test when device.type is not "cuda" or "cpu"
    with pytest.raises(Exception):
        move_to_device(model, torch.device("opengl"))
