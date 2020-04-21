# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import papermill as pm
import pytest
import scrapbook as sb
from tests.notebooks_common import KERNEL_NAME, OUTPUT_NOTEBOOK
import torch

ABS_TOL = 0.02


@pytest.mark.gpu
@pytest.mark.integration
def test_minilm_abstractive_summarization(notebooks, tmp):
    notebook_path = notebooks["minilm_abstractive_summarization"]
    pm.execute_notebook(
        notebook_path,
        OUTPUT_NOTEBOOK,
        kernel_name=KERNEL_NAME,
        parameters=dict(
            QUICK_RUN=True,
            NUM_GPUS=torch.cuda.device_count(),
            TOP_N=100,
            WARMUP_STEPS=5,
            MAX_STEPS=50,
            GRADIENT_ACCUMULATION_STEPS=1,
            TEST_PER_GPU_BATCH_SIZE=2,
            BEAM_SIZE=3,
            CLEANUP_RESULTS=True,
        ),
    )
    result = sb.read_notebook(OUTPUT_NOTEBOOK).scraps.data_dict
    assert pytest.approx(result["rouge_1_f_score"], 0.2, abs=ABS_TOL)
    assert pytest.approx(result["rouge_2_f_score"], 0.07, abs=ABS_TOL)
    assert pytest.approx(result["rouge_l_f_score"], 0.16, abs=ABS_TOL)

@pytest.mark.cpu
@pytest.mark.integration
def test_minilm_abstractive_summarization(notebooks, tmp):
    notebook_path = notebooks["minilm_abstractive_summarization"]
    pm.execute_notebook(
        notebook_path,
        OUTPUT_NOTEBOOK,
        kernel_name=KERNEL_NAME,
        parameters=dict(
            QUICK_RUN=True,
            NUM_GPUS=0,
            TOP_N=2,
            WARMUP_STEPS=5,
            MAX_STEPS=50,
            GRADIENT_ACCUMULATION_STEPS=1,
            TEST_PER_GPU_BATCH_SIZE=2,
            BEAM_SIZE=3,
            CLEANUP_RESULTS=True,
        ),
    )
    result = sb.read_notebook(OUTPUT_NOTEBOOK).scraps.data_dict
    assert pytest.approx(result["rouge_1_f_score"], 0.1, abs=ABS_TOL)
    assert pytest.approx(result["rouge_2_f_score"], 0.05, abs=ABS_TOL)
    assert pytest.approx(result["rouge_l_f_score"], 0.1, abs=ABS_TOL)

