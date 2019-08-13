# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import json
import shutil
import pytest
import papermill as pm
import scrapbook as sb
from tests.notebooks_common import OUTPUT_NOTEBOOK, KERNEL_NAME


ABS_TOL = 0.1


@pytest.mark.gpu
@pytest.mark.integration
def test_tc_mnli_bert(notebooks, tmp):
    notebook_path = notebooks["tc_mnli_bert"]
    pm.execute_notebook(
        notebook_path, 
        OUTPUT_NOTEBOOK, 
        kernel_name=KERNEL_NAME, 
        parameters=dict(NUM_GPUS=1,
                        DATA_FOLDER=tmp,
                        BERT_CACHE_DIR=tmp,
                        BATCH_SIZE=32,
                        BATCH_SIZE_PRED=512,
                        NUM_EPOCHS=1
                       )
    )
    result = sb.read_notebook(OUTPUT_NOTEBOOK).scraps.data_dict
    assert pytest.approx(result["accuracy"], 0.93, abs=ABS_TOL)
    assert pytest.approx(result["precision"], 0.93, abs=ABS_TOL)
    assert pytest.approx(result["recall"], 0.93, abs=ABS_TOL)
    assert pytest.approx(result["f1"], 0.93, abs=ABS_TOL)
    

@pytest.mark.integration
@pytest.mark.azureml
@pytest.mark.gpu
def test_tc_bert_azureml(
    notebooks, subscription_id, resource_group, workspace_name, workspace_region, cluster_name, tmp
):
    notebook_path = notebooks["tc_bert_azureml"]

    train_folder = os.path.join(tmp, "train")
    test_folder = os.path.join(tmp, "test")

    parameters = {
        "config_path": "tests/ci",
        "subscription_id": subscription_id,
        "resource_group": resource_group,
        "workspace_name": workspace_name,
        "workspace_region": workspace_region,
        "cluster_name": cluster_name,
        "DATA_FOLDER": tmp,
        "TRAIN_FOLDER": train_folder,
        "TEST_FOLDER": test_folder,
        "PROJECT_FOLDER": "./",
        "NUM_PARTITIONS": 1,
        "NODE_COUNT": 1,
    }

    pm.execute_notebook(
        notebook_path, OUTPUT_NOTEBOOK, kernel_name=KERNEL_NAME, parameters=parameters
    )

    with open("outputs/results.json", "r") as handle:
        result_dict = json.load(handle)
        assert result_dict["weighted avg"]["f1-score"] == pytest.approx(0.85, abs=ABS_TOL)

    if os.path.exists("outputs"):
        shutil.rmtree("outputs")
