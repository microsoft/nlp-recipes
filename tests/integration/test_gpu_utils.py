# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pytest
import torch


@pytest.mark.gpu
@pytest.mark.integration
def test_machine_is_gpu_machine():
    assert torch.cuda.is_available() is True
