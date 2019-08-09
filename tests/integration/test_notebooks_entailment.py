# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pytest
import papermill as pm
import os
import json
import shutil
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


@pytest.mark.integration
@pytest.mark.azureml
def test_entailment_bert_azureml(notebooks,
                                 subscription_id,
                                 resource_group,
                                 workspace_name,
                                 workspace_region):
    notebook_path = notebooks["entailment_bert_azureml"]
    pm.execute_notebook(notebook_path,
                        OUTPUT_NOTEBOOK,
                        parameters={'DATA_PERCENT_USED': 0.0025,
                                    'subscription_id': subscription_id,
                                    'resource_group': resource_group,
                                    'workspace_name': workspace_name,
                                    'workspace_region': workspace_region},
                        kernel_name=KERNEL_NAME,)

    with open("outputs/results.json", "r") as handle:
        result_dict = json.load(handle)
        assert result_dict["weighted avg"]["f1-score"] > 0.5

    if os.path.exists("outputs"):
        shutil.rmtree("outputs")
