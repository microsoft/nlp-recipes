# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pytest
import papermill as pm
import scrapbook as sb
from tests.notebooks_common import OUTPUT_NOTEBOOK

ABS_TOL = 0.2


@pytest.mark.integration
@pytest.mark.azureml
def test_bidaf_deep_dive(
    notebooks, subscription_id, resource_group, workspace_name, workspace_region
):
    notebook_path = notebooks["bidaf_deep_dive"]
    pm.execute_notebook(
        notebook_path,
        OUTPUT_NOTEBOOK,
        parameters={
            "NUM_EPOCHS": 2,
            "config_path": "tests/ci",
            "PROJECT_FOLDER": "examples/question_answering/bidaf-question-answering",
            "SQUAD_FOLDER": "examples/question_answering/squad",
            "LOGS_FOLDER": "examples/question_answering/",
            "BIDAF_CONFIG_PATH": "examples/question_answering/",
            "subscription_id": subscription_id,
            "resource_group": resource_group,
            "workspace_name": workspace_name,
            "workspace_region": workspace_region,
        },
    )
    result = sb.read_notebook(OUTPUT_NOTEBOOK).scraps.data_dict["validation_EM"]
    assert result == pytest.approx(0.5, abs=ABS_TOL)


@pytest.mark.usefixtures("teardown_service")
@pytest.mark.integration
@pytest.mark.azureml
def test_bidaf_quickstart(
    notebooks, subscription_id, resource_group, workspace_name, workspace_region
):
    notebook_path = notebooks["bidaf_quickstart"]
    pm.execute_notebook(
        notebook_path,
        OUTPUT_NOTEBOOK,
        parameters={
            "config_path": "tests/ci",
            "subscription_id": subscription_id,
            "resource_group": resource_group,
            "workspace_name": workspace_name,
            "workspace_region": workspace_region,
            "webservice_name": "aci-test-service",
        },
    )
    result = sb.read_notebook(OUTPUT_NOTEBOOK).scraps.data_dict["answer"]
    assert result == "Bi-Directional Attention Flow"


@pytest.mark.integration
@pytest.mark.azureml
@pytest.mark.gpu
def test_bert_qa_runs(notebooks):
    notebook_path = notebooks["bert_qa_trainer"]
    pm.execute_notebook(
        notebook_path,
        OUTPUT_NOTEBOOK,
        parameters=dict(
            AZUREML_CONFIG_PATH="./tests/integration/.azureml",
            DATA_FOLDER="./tests/integration/squad",
            PROJECT_FOLDER="./tests/integration/pytorch-transformers",
            EXPERIMENT_NAME="NLP-QA-BERT-deepdive",
            BERT_UTIL_PATH="./utils_nlp/azureml/azureml_bert_util.py",
            EVALUATE_SQAD_PATH="./utils_nlp/eval/evaluate_squad.py",
            TRAIN_SCRIPT_PATH="./examples/question_answering/bert_run_squad_azureml.py",
            BERT_MODEL="bert-base-uncased",
            NUM_TRAIN_EPOCHS=1.0,
            NODE_COUNT=1,
            MAX_TOTAL_RUNS=1,
            MAX_CONCURRENT_RUNS=1,
            TARGET_GRADIENT_STEPS=1,
            INIT_GRADIENT_STEPS=1,
        ),
    )
    result = sb.read_notebook(OUTPUT_NOTEBOOK).scraps.data_dict
    assert result["f1"] > 70
    assert result["learning_rate"] >= 5e-5
    assert result["learning_rate"] <= 9e-5

