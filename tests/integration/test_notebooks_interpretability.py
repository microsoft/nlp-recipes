# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pytest
import numpy as np
import papermill as pm
import scrapbook as sb
from tests.notebooks_common import OUTPUT_NOTEBOOK, KERNEL_NAME


@pytest.mark.gpu
@pytest.mark.integration
def test_deep_and_unified_understanding(notebooks):
    notebook_path = notebooks["deep_and_unified_understanding"]
    pm.execute_notebook(
        notebook_path,
        OUTPUT_NOTEBOOK,
        kernel_name=KERNEL_NAME)
    
    result = sb.read_notebook(OUTPUT_NOTEBOOK).scraps.data_dict
    sigma_numbers = [0.00317593, 0.00172284, 0.00634005, 0.00164305, 0.00317159]
    sigma_bert = [0.1735696 , 0.14028822, 0.14590865, 0.2263149 , 0.20640415,
       0.21249843, 0.18685372, 0.14112663, 0.25824168, 0.22399105,
       0.2393731 , 0.12868434, 0.27386534, 0.35876372]
    
    np.testing.assert_array_almost_equal(result["sigma_numbers"], sigma_numbers, decimal=3) 
    np.testing.assert_array_almost_equal(result["sigma_bert"], sigma_bert, decimal=1) 
    