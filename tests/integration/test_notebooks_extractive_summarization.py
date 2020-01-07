# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import json
import shutil
import pytest
import papermill as pm
import scrapbook as sb
from tests.notebooks_common import OUTPUT_NOTEBOOK, KERNEL_NAME


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
            TOP_N=1000,
            CHUNK_SIZE=200,
            USE_PREPROCESSED_DATA=False,
            NUM_GPUS=1,
            DATA_FOLDER=tmp,
            CACHE_DIR=tmp,
            BATCH_SIZE=3000,
            REPORT_EVERY=50,
            MAX_STEPS=1e3,
            WARMUP_STEPS=5e2,
            MODEL_NAME="distilbert-base-uncased",
        ),
    )
    result = sb.read_notebook(OUTPUT_NOTEBOOK).scraps.data_dict
    print(result)
    assert pytest.approx(result["rouge_2_f_score"], 0.1, abs=ABS_TOL)


