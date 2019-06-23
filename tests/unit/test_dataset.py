# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import pytest

from utils_nlp.dataset.url_utils import maybe_download
from utils_nlp.dataset.msrpc import load_pandas_df
import utils_nlp.dataset.wikigold as wg


def test_maybe_download():
    # ToDo: Change this url when repo goes public.
    file_url = (
        "https://raw.githubusercontent.com/Microsoft/Recommenders/"
        "master/LICENSE"
    )
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


def test_wikigold(tmp_path):
    wg_sentence_count = 1841
    wg_test_percentage = 0.5
    wg_test_sentence_count = round(wg_sentence_count * wg_test_percentage)
    wg_train_sentence_count = wg_sentence_count - wg_test_sentence_count

    downloaded_file = os.path.join(tmp_path, "wikigold.conll.txt")
    assert not os.path.exists(downloaded_file)

    train_df, test_df = wg.load_train_test_dfs(
        tmp_path, test_percentage=wg_test_percentage
    )

    assert os.path.exists(downloaded_file)

    assert train_df.shape == (wg_train_sentence_count, 2)
    assert test_df.shape == (wg_test_sentence_count, 2)
