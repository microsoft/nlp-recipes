# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pytest
import papermill as pm
import scrapbook as sb
import os
import json
import shutil
from tests.notebooks_common import OUTPUT_NOTEBOOK, KERNEL_NAME

ABS_TOL = 0.1


@pytest.mark.gpu
@pytest.mark.integration
def test_entailment_multinli_bert(notebooks, tmp):
    notebook_path = notebooks["entailment_multinli_transformers"]
    pm.execute_notebook(
        notebook_path,
        OUTPUT_NOTEBOOK,
        parameters={
            "MODEL_NAME": "bert-base-uncased",
            "TO_LOWER": True,
            "TRAIN_DATA_USED_FRACTION": 0.05,
            "DEV_DATA_USED_FRACTION": 0.05,
            "NUM_EPOCHS": 1,
            "CACHE_DIR": tmp
        },
        kernel_name=KERNEL_NAME,
    )
    result = sb.read_notebook(OUTPUT_NOTEBOOK).scraps.data_dict
    assert pytest.approx(result["matched_precision"], 0.76, abs=ABS_TOL)
    assert pytest.approx(result["matched_recall"], 0.76, abs=ABS_TOL)
    assert pytest.approx(result["matched_f1"], 0.76, abs=ABS_TOL)
    assert pytest.approx(result["mismatched_precision"], 0.76, abs=ABS_TOL)
    assert pytest.approx(result["mismatched_recall"], 0.76, abs=ABS_TOL)
    assert pytest.approx(result["mismatched_f1"], 0.76, abs=ABS_TOL)

@pytest.mark.integration
@pytest.mark.azureml
def test_entailment_xnli_bert_azureml(
    notebooks, subscription_id, resource_group, workspace_name, workspace_region, cluster_name
):
    notebook_path = notebooks["entailment_xnli_bert_azureml"]
    pm.execute_notebook(
        notebook_path,
        OUTPUT_NOTEBOOK,
        parameters={
            "DATA_PERCENT_USED": 0.0025,
            "subscription_id": subscription_id,
            "resource_group": resource_group,
            "workspace_name": workspace_name,
            "workspace_region": workspace_region,
            "cluster_name": cluster_name,
        },
        kernel_name=KERNEL_NAME,
    )

    with open("outputs/results.json", "r") as handle:
        result_dict = json.load(handle)
        assert result_dict["weighted avg"]["f1-score"] == pytest.approx(0.2, abs=ABS_TOL)

    if os.path.exists("outputs"):
        shutil.rmtree("outputs")
