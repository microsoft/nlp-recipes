# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import pytest

from utils_nlp.dataset import msrpc


@pytest.mark.smoke
def test_msrpc_download(tmp_path):
    filepath = msrpc.download_msrpc(tmp_path)
    statinfo = os.stat(filepath)
    assert statinfo.st_size == 1359872


@pytest.mark.skip(reason="Can't test it programmatically, needs input")
@pytest.mark.smoke
def test_msrpc_load_df(tmp_path):
    df_train = msrpc.load_pandas_df(
        local_cache_path=tmp_path, dataset_type="train"
    )
