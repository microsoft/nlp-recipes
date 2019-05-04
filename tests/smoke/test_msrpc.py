# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import pytest

from utils_nlp.dataset import msrpc


@pytest.mark.smoke
def test_download_msrpc(tmp_path):
    filepath = msrpc.download_msrpc(tmp_path)
    statinfo = os.stat(filepath)
    assert statinfo.st_size == 1359872
