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
def test_tc_mnli_transformers(notebooks, tmp):
    notebook_path = notebooks["tc_mnli_transformers"]
    pm.execute_notebook(
        notebook_path,
        OUTPUT_NOTEBOOK,
        kernel_name=KERNEL_NAME,
        parameters=dict(
            NUM_GPUS=1,
            DATA_FOLDER=tmp,
            CACHE_DIR=tmp,
            BATCH_SIZE=16,
            NUM_EPOCHS=1,
            TRAIN_DATA_FRACTION=0.05,
            TEST_DATA_FRACTION=0.05,
            MODEL_NAMES=["distilbert-base-uncased"],
        ),
    )
    result = sb.read_notebook(OUTPUT_NOTEBOOK).scraps.data_dict
    assert pytest.approx(result["accuracy"], 0.885, abs=ABS_TOL)
    assert pytest.approx(result["f1"], 0.885, abs=ABS_TOL)


@pytest.mark.integration
@pytest.mark.azureml
@pytest.mark.gpu
def test_tc_bert_azureml(
    notebooks, subscription_id, resource_group, workspace_name, workspace_region, tmp
):
    notebook_path = notebooks["tc_bert_azureml"]

    train_folder = os.path.join(tmp, "train")
    test_folder = os.path.join(tmp, "test")

    parameters = {
        "config_path": None,
        "subscription_id": subscription_id,
        "resource_group": resource_group,
        "workspace_name": workspace_name,
        "workspace_region": workspace_region,
        "cluster_name": "tc-bert-cluster",
        "DATA_FOLDER": tmp,
        "TRAIN_FOLDER": train_folder,
        "TEST_FOLDER": test_folder,
        "PROJECT_FOLDER": ".",
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


@pytest.mark.gpu
@pytest.mark.integration
def test_multi_languages_transformer(notebooks, tmp):
    notebook_path = notebooks["tc_multi_languages_transformers"]
    pm.execute_notebook(
        notebook_path,
        OUTPUT_NOTEBOOK,
        kernel_name=KERNEL_NAME,
        parameters={"QUICK_RUN": True, "USE_DATASET": "dac"},
    )
    result = sb.read_notebook(OUTPUT_NOTEBOOK).scraps.data_dict
    assert pytest.approx(result["precision"], 0.94, abs=ABS_TOL)
    assert pytest.approx(result["recall"], 0.94, abs=ABS_TOL)
    assert pytest.approx(result["f1"], 0.94, abs=ABS_TOL)
