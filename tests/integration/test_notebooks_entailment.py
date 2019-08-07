# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pytest
import papermill as pm
from tests.notebooks_common import OUTPUT_NOTEBOOK, KERNEL_NAME


@pytest.mark.gpu
@pytest.mark.integration
def test_entailment_multinli_bert(notebooks):
    notebook_path = notebooks["entailment_multinli_bert"]
    pm.execute_notebook(
        notebook_path,
        OUTPUT_NOTEBOOK,
        parameters={
            "TRAIN_DATA_USED_PERCENT": 0.001,
            "DEV_DATA_USED_PERCENT": 0.01,
            "NUM_EPOCHS": 1,
        },
        kernel_name=KERNEL_NAME,
    )
