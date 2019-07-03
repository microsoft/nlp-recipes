# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import sys
import pytest
import papermill as pm
import scrapbook as sb
from azureml.core import Experiment
from azureml.core.run import Run

sys.path.append("../../")
from utils_nlp.azureml.azureml_utils import get_or_create_workspace
from tests.notebooks_common import OUTPUT_NOTEBOOK


ABS_TOL = 0.1


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