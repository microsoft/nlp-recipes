# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os

import pytest

from utils_nlp.dataset.url_utils import maybe_download
from utils_nlp.dataset.msrpc import load_pandas_df


def test_maybe_download():
    # ToDo: Change this url when repo goes public.
    file_url = "https://raw.githubusercontent.com/Microsoft/Recommenders/master/LICENSE"
    filepath = "license.txt"
    assert not os.path.exists(filepath)
    filepath = maybe_download(file_url, "license.txt", expected_bytes=1162)
    assert os.path.exists(filepath)
    os.remove(filepath)
    with pytest.raises(IOError):
        filepath = maybe_download(file_url, "license.txt", expected_bytes=0)


def test_load_pandas_df_msrpc():
    with pytest.raises(Exception):
        load_pandas_df(dataset_type="Dummy")
