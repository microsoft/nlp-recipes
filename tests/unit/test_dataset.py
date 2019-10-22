# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import pytest

from utils_nlp.dataset.url_utils import maybe_download
from utils_nlp.dataset import msrpc
from utils_nlp.dataset import wikigold
from utils_nlp.dataset import xnli
from utils_nlp.dataset import snli
from utils_nlp.dataset import Split
from utils_nlp.dataset import squad
from utils_nlp.dataset.ner_utils import preprocess_conll


@pytest.fixture
def ner_utils_test_data(scope="module"):
    return {
        "input": "The O\n139th I-ORG\nwas O\nformed O\nat O\nCamp I-LOC\n"
        "Howe I-LOC\n, O\nnear O\nPittsburgh I-LOC\n, O\non O\n"
        "September O\n1 O\n, O\n1862 O\n. O\n\nFrederick I-PER\n"
        "H. I-PER\nCollier I-PER\nwas O\nthe O\nfirst O\ncolonel O\n. O",
        "expected_output": (
            [
                [
                    "The",
                    "139th",
                    "was",
                    "formed",
                    "at",
                    "Camp",
                    "Howe",
                    ",",
                    "near",
                    "Pittsburgh",
                    ",",
                    "on",
                    "September",
                    "1",
                    ",",
                    "1862",
                    ".",
                ],
                ["Frederick", "H.", "Collier", "was", "the", "first", "colonel", "."],
            ],
            [
                [
                    "O",
                    "I-ORG",
                    "O",
                    "O",
                    "O",
                    "I-LOC",
                    "I-LOC",
                    "O",
                    "O",
                    "I-LOC",
                    "O",
                    "O",
                    "O",
                    "O",
                    "O",
                    "O",
                    "O",
                ],
                ["I-PER", "I-PER", "I-PER", "O", "O", "O", "O", "O"],
            ],
        ),
    }


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


def test_msrpc():
    with pytest.raises(Exception):
        msrpc.load_pandas_df(dataset_type="Dummy")


def test_wikigold(tmp_path):
    wg_sentence_count = 1841
    wg_test_fraction = 0.5
    wg_test_sentence_count = round(wg_sentence_count * wg_test_fraction)
    wg_train_sentence_count = wg_sentence_count - wg_test_sentence_count

    downloaded_file = os.path.join(tmp_path, "wikigold.conll.txt")
    assert not os.path.exists(downloaded_file)

    train_df, test_df = wikigold.load_train_test_dfs(tmp_path, test_fraction=wg_test_fraction)

    assert os.path.exists(downloaded_file)

    assert train_df.shape == (wg_train_sentence_count, 2)
    assert test_df.shape == (wg_test_sentence_count, 2)


def test_ner_utils(ner_utils_test_data):
    output = preprocess_conll(ner_utils_test_data["input"])
    assert output == ner_utils_test_data["expected_output"]


def test_xnli(tmp_path):
    # Only test for the dev df as the train dataset takes several
    # minutes to download
    dev_df = xnli.load_pandas_df(local_cache_path=tmp_path, file_split="dev")
    assert dev_df.shape == (2490, 2)


def test_snli(tmp_path):
    df_train = snli.load_pandas_df(local_cache_path=tmp_path, file_split=Split.TRAIN)
    assert df_train.shape == (550152, 14)
    df_test = snli.load_pandas_df(local_cache_path=tmp_path, file_split=Split.TEST)
    assert df_test.shape == (10000, 14)
    df_dev = snli.load_pandas_df(local_cache_path=tmp_path, file_split=Split.DEV)
    assert df_dev.shape == (10000, 14)


def test_squad(tmp_path):
    v1_train_df = squad.load_pandas_df(
        local_cache_path=tmp_path, squad_version="v1.1", file_split="train"
    )
    assert v1_train_df.shape == (87599, 6)

    v1_dev_df = squad.load_pandas_df(
        local_cache_path=tmp_path, squad_version="v1.1", file_split="dev"
    )
    assert v1_dev_df.shape == (10570, 6)

    v2_train_df = squad.load_pandas_df(
        local_cache_path=tmp_path, squad_version="v2.0", file_split="train"
    )
    assert v2_train_df.shape == (130319, 6)

    v2_dev_df = squad.load_pandas_df(
        local_cache_path=tmp_path, squad_version="v2.0", file_split="dev"
    )
    assert v2_dev_df.shape == (11873, 6)
