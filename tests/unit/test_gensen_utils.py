# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os

import pandas as pd

from utils_nlp.models.gensen.preprocess_utils import gensen_preprocess
from utils_nlp.models.gensen.utils import DataIterator


def test_gensen_preprocess(tmp_path):
    data = [
        [
            "neutral",
            "it is a lovely day",
            "the weather is great outside.",
            ["it", "is", "lovely", "day"],
            ["the", "weather", "is", "great", "outside"],
        ]
    ]

    df = pd.DataFrame(data)
    df.columns = [
        "score",
        "sentence1",
        "sentence2",
        "sentence1_tokens",
        "sentence2_tokens",
    ]

    expected_files = [
        "snli_1.0_test.txt.lab",
        "snli_1.0_test.txt.s1.tok",
        "snli_1.0_dev.txt.clean.noblank",
        "snli_1.0_train.txt.s1.tok",
        "snli_1.0_train.txt.lab",
        "snli_1.0_dev.txt.s1.tok",
        "snli_1.0_dev.txt.s2.tok",
        "snli_1.0_test.txt.s2.tok",
        "snli_1.0_train.txt.clean",
        "snli_1.0_train.txt.s2.tok",
        "snli_1.0_test.txt.clean.noblank",
        "snli_1.0_test.txt.clean",
        "snli_1.0_train.txt.clean.noblank",
        "snli_1.0_dev.txt.lab",
        "snli_1.0_dev.txt.clean",
    ]
    path = gensen_preprocess(df, df, df, tmp_path)
    assert os.path.isdir(path) is True
    assert set(os.listdir(path)) == set(expected_files)


def test_data_iterator():
    sentences = ["it is a lovely day", "the weather is great outside.", ]
    expected_vocab = ["it", "is", "a", "lovely", "day", "the", "weather", "is", "great", "outside."]

    vocab_size = 10
    di = DataIterator()
    word2id, id2word = di.construct_vocab(sentences, vocab_size)
    assert set(expected_vocab).issubset(word2id.keys())
    assert set(expected_vocab).issubset(id2word.values())
