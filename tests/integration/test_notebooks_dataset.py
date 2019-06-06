# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pytest
import papermill as pm

from tests.notebooks_common import OUTPUT_NOTEBOOK, KERNEL_NAME


@pytest.mark.notebooks
@pytest.mark.skip(reason="no way of running this programmatically")
def test_msrpc_runs(notebooks):
    notebook_path = notebooks["msrpc"]
    pm.execute_notebook(
        notebook_path,
        OUTPUT_NOTEBOOK,
        kernel_name=KERNEL_NAME,
    )

@pytest.mark.notebooks
def test_snli_runs(notebooks):
    notebook_path = notebooks["snli"]
    pm.execute_notebook(
        notebook_path,
        OUTPUT_NOTEBOOK,
        kernel_name=KERNEL_NAME,
    )

@pytest.mark.notebooks
def test_stsbenchmark_runs(notebooks):
    notebook_path = notebooks["stsbenchmark"]
    pm.execute_notebook(
        notebook_path,
        OUTPUT_NOTEBOOK,
        kernel_name=KERNEL_NAME,
    )