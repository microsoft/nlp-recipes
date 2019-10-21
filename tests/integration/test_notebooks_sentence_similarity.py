# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pytest
import papermill as pm
import scrapbook as sb
from tests.notebooks_common import OUTPUT_NOTEBOOK, KERNEL_NAME


ABS_TOL = 0.2
ABS_TOL_PEARSONS = 0.05


@pytest.fixture(scope="module")
def baseline_results():
    return {
        "Word2vec Cosine": 0.6476606845766778,
        "Word2vec Cosine with Stop Words": 0.6683808069062863,
        "Word2vec WMD": 0.6574175839579567,
        "Word2vec WMD with Stop Words": 0.6574175839579567,
        "GLoVe Cosine": 0.6688056947022161,
        "GLoVe Cosine with Stop Words": 0.6049380247374541,
        "GLoVe WMD": 0.6267300417407605,
        "GLoVe WMD with Stop Words": 0.48470008225931194,
        "fastText Cosine": 0.6707510007525627,
        "fastText Cosine with Stop Words": 0.6771300330824099,
        "fastText WMD": 0.6394958913339955,
        "fastText WMD with Stop Words": 0.5177829727556036,
        "TF-IDF Cosine": 0.6749213786510483,
        "TF-IDF Cosine with Stop Words": 0.7118087132257667,
        "Doc2vec Cosine": 0.5337078384749167,
        "Doc2vec Cosine with Stop Words": 0.4498543211602068,
    }


@pytest.mark.integration
def test_similarity_embeddings_baseline_runs(notebooks, baseline_results):
    notebook_path = notebooks["similarity_embeddings_baseline"]
    pm.execute_notebook(notebook_path, OUTPUT_NOTEBOOK, kernel_name=KERNEL_NAME)
    results = sb.read_notebook(OUTPUT_NOTEBOOK).scraps.data_dict["results"]
    for key, value in baseline_results.items():
        assert results[key] == pytest.approx(value, abs=ABS_TOL)


@pytest.mark.gpu
@pytest.mark.integration
@pytest.mark.skip(
    reason="push for release, no horovod installation automation or documentation yet"
)
def test_gensen_local(notebooks):
    notebook_path = notebooks["gensen_local"]
    pm.execute_notebook(
        notebook_path,
        OUTPUT_NOTEBOOK,
        kernel_name=KERNEL_NAME,
        parameters=dict(
            max_epoch=1,
            config_filepath="examples/sentence_similarity/gensen_config.json",
            base_data_path="data",
        ),
    )

    results = sb.read_notebook(OUTPUT_NOTEBOOK).scraps.data_dict["results"]
    expected = {"0": {"0": 1, "1": 0.95}, "1": {"0": 0.95, "1": 1}}

    for key, value in expected.items():
        for k, v in value.items():
            assert results[key][k] == pytest.approx(v, abs=ABS_TOL_PEARSONS)


@pytest.mark.gpu
@pytest.mark.integration
def test_bert_encoder(notebooks, tmp):
    notebook_path = notebooks["bert_encoder"]
    pm.execute_notebook(
        notebook_path,
        OUTPUT_NOTEBOOK,
        kernel_name=KERNEL_NAME,
        parameters=dict(NUM_GPUS=1, MAX_SEQ_LENGTH=128, CACHE_DIR=tmp),
    )
    size_emb = sb.read_notebook(OUTPUT_NOTEBOOK).scraps.data_dict["size_emb"]
    assert size_emb == 768


@pytest.mark.integration
@pytest.mark.azureml
def test_bert_senteval(
    notebooks, subscription_id, resource_group, workspace_name, workspace_region, tmp
):
    notebook_path = notebooks["bert_senteval"]
    pm.execute_notebook(
        notebook_path,
        OUTPUT_NOTEBOOK,
        kernel_name=KERNEL_NAME,
        parameters=dict(
            subscription_id=subscription_id,
            resource_group=resource_group,
            workspace_name=workspace_name,
            workspace_region=workspace_region,
            CACHE_DIR=tmp,
            LOCAL_UTILS="utils_nlp",
            LOCAL_SENTEVAL="utils_nlp/eval/SentEval",
            EXPERIMENT_NAME="test-nlp-ss-bert",
            CLUSTER_NAME="eval-gpu",
            MAX_NODES=1,
        ),
    )
    pearson = sb.read_notebook(OUTPUT_NOTEBOOK).scraps.data_dict["pearson"]
    mse = sb.read_notebook(OUTPUT_NOTEBOOK).scraps.data_dict["mse"]
    assert pearson == pytest.approx(0.6, abs=ABS_TOL)
    assert mse < 1.8


@pytest.mark.integration
@pytest.mark.azureml
def test_similarity_embeddings_baseline_runs(notebooks, baseline_results):
    notebook_path = notebooks["similarity_embeddings_baseline"]
    pm.execute_notebook(notebook_path, OUTPUT_NOTEBOOK, kernel_name=KERNEL_NAME)
    results = sb.read_notebook(OUTPUT_NOTEBOOK).scraps.data_dict["results"]
    for key, value in baseline_results.items():
        assert results[key] == pytest.approx(value, abs=ABS_TOL)


@pytest.mark.usefixtures("teardown_service")
@pytest.mark.integration
@pytest.mark.azureml
def test_automl_local_deployment_aci(
    notebooks, subscription_id, resource_group, workspace_name, workspace_region
):
    notebook_path = notebooks["automl_local_deployment_aci"]
    pm.execute_notebook(
        notebook_path,
        OUTPUT_NOTEBOOK,
        parameters={
            "automl_iterations": 1,
            "automl_iteration_timeout": 7,
            "config_path": None,
            "webservice_name": "aci-test-service",
            "subscription_id": subscription_id,
            "resource_group": resource_group,
            "workspace_name": workspace_name,
            "workspace_region": workspace_region,
        },
    )
    result = sb.read_notebook(OUTPUT_NOTEBOOK).scraps.data_dict["pearson_correlation"]
    assert result == pytest.approx(0.5, abs=ABS_TOL)


@pytest.mark.integration
@pytest.mark.azureml
@pytest.mark.skip(
    reason="push for release, no horovod installation automation or documentation yet"
)
def test_gensen_aml_deep_dive(notebooks):
    notebook_path = notebooks["gensen_aml_deep_dive"]
    pm.execute_notebook(
        notebook_path,
        OUTPUT_NOTEBOOK,
        parameters=dict(
            CACHE_DIR="./tests/integration/temp",
            AZUREML_CONFIG_PATH="./tests/integration/.azureml",
            UTIL_NLP_PATH="./utils_nlp",
            MAX_EPOCH=1,
            TRAIN_SCRIPT="./examples/sentence_similarity/gensen_train.py",
            CONFIG_PATH="./examples/sentence_similarity/gensen_config.json",
            MAX_TOTAL_RUNS=1,
            MAX_CONCURRENT_RUNS=1,
        ),
    )
    result = sb.read_notebook(OUTPUT_NOTEBOOK).scraps.data_dict
    assert result["min_val_loss"] > 5
    assert result["learning_rate"] >= 0.0001
    assert result["learning_rate"] <= 0.001


@pytest.mark.integration
@pytest.mark.azureml
@pytest.mark.skip(
    reason="can't run programmatically, AKS cluster takes ~20 minutes to create and there is no blocking call in the notebook to tell that the cluster creation is in progress"
)
def test_automl_with_pipelines_deployment_aks(notebooks):
    notebook_path = notebooks["automl_with_pipelines_deployment_aks"]
    pm.execute_notebook(notebook_path, OUTPUT_NOTEBOOK)
