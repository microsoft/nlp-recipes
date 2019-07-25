# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import sys
import pytest
import papermill as pm
import scrapbook as sb
from azureml.core import Experiment
from azureml.core.run import Run
from utils_nlp.azureml.azureml_utils import get_or_create_workspace
from tests.notebooks_common import OUTPUT_NOTEBOOK

sys.path.append("../../")
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


@pytest.mark.notebooks
def test_similarity_embeddings_baseline_runs(notebooks, baseline_results):
    notebook_path = notebooks["similarity_embeddings_baseline"]
    pm.execute_notebook(notebook_path, OUTPUT_NOTEBOOK)
    results = sb.read_notebook(OUTPUT_NOTEBOOK).scraps.data_dict["results"]
    for key, value in baseline_results.items():
        assert results[key] == pytest.approx(value, abs=ABS_TOL)


@pytest.mark.notebooks
@pytest.mark.gpu
def test_similarity_senteval_local_runs(notebooks, gensen_senteval_results):
    notebook_path = notebooks["senteval_local"]
    pm.execute_notebook(
        notebook_path,
        OUTPUT_NOTEBOOK,
        parameters=dict(
            PATH_TO_SENTEVAL="../SentEval", PATH_TO_GENSEN="../gensen"
        ),
    )
    out = sb.read_notebook(OUTPUT_NOTEBOOK).scraps.data_dict["results"]
    for key, val in gensen_senteval_results.items():
        for task, result in val.items():
            assert out[key][task] == result


@pytest.mark.notebooks
@pytest.mark.azureml
def test_similarity_senteval_azureml_runs(notebooks, gensen_senteval_results):
    notebook_path = notebooks["senteval_azureml"]
    pm.execute_notebook(
        notebook_path,
        OUTPUT_NOTEBOOK,
        parameters=dict(
            PATH_TO_SENTEVAL="../SentEval",
            PATH_TO_GENSEN="../gensen",
            PATH_TO_SER="utils_nlp/eval/senteval.py",
            AZUREML_VERBOSE=False,
            config_path="tests/ci",
        ),
    )
    result = sb.read_notebook(OUTPUT_NOTEBOOK).scraps.data_dict
    ws = get_or_create_workspace(config_path="tests/ci")
    experiment = Experiment(ws, name=result["experiment_name"])
    run = Run(experiment, result["run_id"])
    assert run.get_metrics()["STSBenchmark::pearson"] == pytest.approx(
        gensen_senteval_results["pearson"]["STSBenchmark"], abs=ABS_TOL
    )


@pytest.mark.notebooks
@pytest.mark.gpu
def test_gensen_local(notebooks):
    notebook_path = notebooks["gensen_local"]
    pm.execute_notebook(
        notebook_path,
        OUTPUT_NOTEBOOK,
        parameters=dict(
            max_epoch=1,
            config_filepath="../../scenarios/sentence_similarity/gensen_config.json",
            base_data_path="../../data",
        ),
    )

    results = sb.read_notebook(OUTPUT_NOTEBOOK).scraps.data_dict["results"]
    expected = {"0": {"0": 1, "1": 0.95}, "1": {"0": 0.95, "1": 1}}

    for key, value in expected.items():
        for k, v in value.items():
            assert results[key][k] == pytest.approx(v, abs=ABS_TOL_PEARSONS)
