# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pytest
import papermill as pm
import scrapbook as sb
from tests.notebooks_common import OUTPUT_NOTEBOOK, KERNEL_NAME

ABS_TOL = 0.05

@pytest.mark.gpu
@pytest.mark.integration
def test_ner_wikigold_bert(notebooks, tmp):
    notebook_path = notebooks["ner_wikigold_transformer"]
    pm.execute_notebook(
        notebook_path,
        OUTPUT_NOTEBOOK,
        parameters={
            "DATA_PATH": tmp,
            "CACHE_DIR": tmp
        },
        kernel_name=KERNEL_NAME,
    )
    result = sb.read_notebook(OUTPUT_NOTEBOOK).scraps.data_dict
    assert pytest.approx(result["precision"], 0.80, abs=ABS_TOL)
    assert pytest.approx(result["recall"], 0.83, abs=ABS_TOL)
    assert pytest.approx(result["f1"], 0.83, abs=ABS_TOL)