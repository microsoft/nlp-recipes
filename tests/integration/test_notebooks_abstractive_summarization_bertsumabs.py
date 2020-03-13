# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import papermill as pm
import pytest
import scrapbook as sb
from tests.notebooks_common import KERNEL_NAME, OUTPUT_NOTEBOOK
import torch

ABS_TOL = 0.02


@pytest.mark.gpu
@pytest.mark.integration
def test_abstractive_summarization_bertsumabs_cnndm(notebooks, tmp):
    notebook_path = notebooks["abstractive_summarization_bertsumabs_cnndm"]
    pm.execute_notebook(
        notebook_path,
        OUTPUT_NOTEBOOK,
        kernel_name=KERNEL_NAME,
        parameters=dict(
            QUICK_RUN=True,
            TOP_N=1000,
            MAX_POS=512,
            DATA_FOLDER=tmp,
            CACHE_DIR=tmp,
            BATCH_SIZE_PER_GPU=3,
            REPORT_EVERY=50,
            MAX_STEPS=100,
            MODEL_NAME="bert-base-uncased",
        ),
    )
    result = sb.read_notebook(OUTPUT_NOTEBOOK).scraps.data_dict
    assert pytest.approx(result["rouge_2_f_score"], 0.01, abs=ABS_TOL)
