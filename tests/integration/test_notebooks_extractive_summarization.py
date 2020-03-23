# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import papermill as pm
import pytest
import scrapbook as sb
from tests.notebooks_common import KERNEL_NAME, OUTPUT_NOTEBOOK

ABS_TOL = 0.02


@pytest.mark.gpu
@pytest.mark.integration
def test_extractive_summarization_cnndm_transformers(notebooks, tmp):
    notebook_path = notebooks["extractive_summarization_cnndm_transformer"]
    pm.execute_notebook(
        notebook_path,
        OUTPUT_NOTEBOOK,
        kernel_name=KERNEL_NAME,
        parameters=dict(
            QUICK_RUN=True,
            TOP_N=100,
            CHUNK_SIZE=200,
            USE_PREPROCESSED_DATA=False,
            DATA_PATH=tmp,
            CACHE_DIR=tmp,
            BATCH_SIZE=3000,
            REPORT_EVERY=50,
            MAX_STEPS=100,
            WARMUP_STEPS=5e2,
            MODEL_NAME="distilbert-base-uncased",
        ),
    )
    result = sb.read_notebook(OUTPUT_NOTEBOOK).scraps.data_dict
    assert pytest.approx(result["rouge_2_f_score"], 0.1, abs=ABS_TOL)


@pytest.mark.skip(reason="no need to test")
@pytest.mark.gpu
@pytest.mark.integration
def test_extractive_summarization_cnndm_transformers_processed(notebooks, tmp):
    notebook_path = notebooks["extractive_summarization_cnndm_transformer"]
    pm.execute_notebook(
        notebook_path,
        OUTPUT_NOTEBOOK,
        kernel_name=KERNEL_NAME,
        parameters=dict(
            QUICK_RUN=True,
            TOP_N=100,
            CHUNK_SIZE=200,
            USE_PREPROCESSED_DATA=True,
            DATA_PATH=tmp,
            CACHE_DIR=tmp,
            PROCESSED_DATA_PATH=tmp,
            BATCH_SIZE=3000,
            REPORT_EVERY=50,
            MAX_STEPS=100,
            WARMUP_STEPS=5e2,
            MODEL_NAME="distilbert-base-uncased",
        ),
    )
    result = sb.read_notebook(OUTPUT_NOTEBOOK).scraps.data_dict
    assert pytest.approx(result["rouge_2_f_score"], 0.1, abs=ABS_TOL)
