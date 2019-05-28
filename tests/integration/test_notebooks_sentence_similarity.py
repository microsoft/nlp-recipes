# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pytest
import papermill as pm

from tests.notebooks_common import OUTPUT_NOTEBOOK, KERNEL_NAME


ABS_TOL = 0.1


@pytest.fixture(scope="module")
def embeddins_results():
    return {
        "Word2vec Cosine": 0.6337760059182685,
        "Word2vec Cosine with Stop Words": 0.647674307797345,
        "Word2vec WMD": 0.6578256301323717,
        "Word2vec WMD with Stop Words": 0.5697910628727217,
        "GLoVe Cosine": 0.642064729899729,
        "GLoVe Cosine with Stop Words": 0.5639670964748242,
        "GLoVe WMD": 0.6272339050920003,
        "GLoVe WMD with Stop Words": 0.48560149551724,
        "fastText Cosine": 0.6288780924569853,
        "fastText Cosine with Stop Words": 0.5958470751204786,
        "fastText WMD": 0.527520845792085,
        "fastText WMD with Stop Words": 0.44198752510004097,
        "TF-IDF Cosine": 0.6683811410442562,
        "TF-IDF Cosine with Stop Words": 0.7034695168223282,
        "Doc2vec Cosine": 0.5082738341805563,
        "Doc2vec Cosine with Stop Words": 0.4116873013460912
 }


@pytest.mark.notebooks
def test_similarity_embeddings_baseline_runs(notebooks, embeddins_results):
    notebook_path = notebooks["similarity_embeddings_baseline"]
    pm.execute_notebook(
        notebook_path,
        OUTPUT_NOTEBOOK,
        kernel_name=KERNEL_NAME,
    )
    metrics = pm.read_notebook(OUTPUT_NOTEBOOK).dataframe.set_index("name")["value"]
    results = metrics["results"]
    for key, value in embeddins_results.items():
        assert results[key] == pytest.approx(value, abs=ABS_TOL)

