# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import pytest
from tests.notebooks_common import OUTPUT_NOTEBOOK, KERNEL_NAME
import papermill as pm
from utils_nlp.models.bert.common import Language


@pytest.mark.notebooks
@pytest.mark.gpu
def test_bert_encoder(notebooks, tmp):
    notebook_path = notebooks["bert_encoder"]
    pm.execute_notebook(
        notebook_path,
        OUTPUT_NOTEBOOK,
        kernel_name=KERNEL_NAME,
        parameters=dict(
            NUM_GPUS=1, LANGUAGE=Language.ENGLISH, TO_LOWER=True, MAX_SEQ_LENGTH=128, CACHE_DIR=tmp
        ),
    )
