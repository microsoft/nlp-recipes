# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np

from utils_nlp.eval.classification import compute_correlation_coefficients


def test_compute():
    x = np.random.rand(2, 100)
    df = compute_correlation_coefficients(x)
    assert df.shape == (2, 2)

    y = np.random.rand(2, 100)
    df = compute_correlation_coefficients(x, y)
    assert df.shape == (4, 4)
