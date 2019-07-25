# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import sys
import pytest
import papermill as pm
import scrapbook as sb

from tests.notebooks_common import OUTPUT_NOTEBOOK


ABS_TOL = 0.2


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
    pm.execute_notebook(notebook_path, OUTPUT_NOTEBOOK)
    results = sb.read_notebook(OUTPUT_NOTEBOOK).scraps.data_dict["results"]
    for key, value in baseline_results.items():
        assert results[key] == pytest.approx(value, abs=ABS_TOL)

