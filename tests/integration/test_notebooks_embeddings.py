# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pytest
import papermill as pm

from tests.notebooks_common import OUTPUT_NOTEBOOK, KERNEL_NAME

@pytest.mark.integration
@pytest.mark.skip(reason="")
@pytest.mark.notebooks
def test_embedding_trainer_runs(notebooks):
    notebook_path = notebooks["embedding_trainer"]
    pm.execute_notebook(
        notebook_path,
        OUTPUT_NOTEBOOK,
        kernel_name=KERNEL_NAME,
        parameters=dict(NLP_REPO_PATH=".")
    )
