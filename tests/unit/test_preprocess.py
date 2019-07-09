# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pytest
import pandas as pd
import numpy as np

import utils_nlp.dataset.preprocess as preprocess


@pytest.fixture(scope="module")
def df_sentences():
    sentences = np.array(
        [
            "The man is playing the piano.",
            "Some men are fighting.",
            "A man is spreading shreded cheese on a pizza.",
            "A man is playing the cello.",
            "A man is spreading shreded cheese on a pizza.",
            "A man is playing a large flute.",
            "A man is playing the cello.",
            "A man is playing on a guitar and singing.",
            "The man is playing the piano.",
            "Some men are fighting.",
        ]
    ).reshape(2, 5)

    return pd.DataFrame(sentences, columns=["s1", "s2", "s3", "s4", "s5"])


def test_to_lowercase_all(df_sentences):
    ldf = preprocess.to_lowercase_all(df_sentences)
    assert sum(map(lambda x: x.islower(), ldf.values.flatten())) == len(
        ldf.values.flatten()
    )


def test_to_lowercase_subset(df_sentences):
    ldf = preprocess.to_lowercase(df_sentences, column_names=["s4"])
    assert sum(map(lambda x: x.islower(), ldf.s4.values.flatten())) == len(
        ldf.s4.values.flatten()
    )


def test_to_spacy_tokens(df_sentences):
    sentence_cols = ["s1", "s2"]
    token_cols = ["t1", "t2"]
    token_df = preprocess.to_spacy_tokens(
        df_sentences, sentence_cols=sentence_cols, token_cols=token_cols
    )
    assert token_df.shape[1] == df_sentences.shape[1] + len(
        token_cols
    ) and sum(
        list(
            map(lambda x: (token_df[x].apply(type) == list).all(), token_cols)
        )
    ) == len(
        token_cols
    )


def test_rm_spacy_stopwords(df_sentences):
    sentence_cols = ["s1", "s2"]
    stop_cols = ["stop1", "stop2"]
    stop_df = preprocess.rm_spacy_stopwords(
        df_sentences, sentence_cols=sentence_cols, stop_cols=stop_cols
    )
    assert stop_df.shape[1] == df_sentences.shape[1] + len(stop_cols) and sum(
        list(map(lambda x: (stop_df[x].apply(type) == list).all(), stop_cols))
    ) == len(stop_cols)


def test_to_nltk_tokens(df_sentences):
    sentence_cols = ["s1", "s2"]
    token_cols = ["t1", "t2"]
    token_df = preprocess.to_nltk_tokens(
        df_sentences, sentence_cols=sentence_cols, token_cols=token_cols
    )
    assert token_df.shape[1] == df_sentences.shape[1] + len(
        token_cols
    ) and sum(
        list(
            map(lambda x: (token_df[x].apply(type) == list).all(), token_cols)
        )
    ) == len(
        token_cols
    )


def test_rm_nltk_stopwords(df_sentences):
    sentence_cols = ["s1", "s2"]
    stop_cols = ["stop1", "stop2"]
    stop_df = preprocess.rm_nltk_stopwords(
        df_sentences, sentence_cols=sentence_cols, stop_cols=stop_cols
    )
    assert stop_df.shape[1] == df_sentences.shape[1] + len(stop_cols) and sum(
        list(map(lambda x: (stop_df[x].apply(type) == list).all(), stop_cols))
    ) == len(stop_cols)


def test_convert_to_unicode():
    test_str = "test"
    test_byte = test_str.encode("utf-8")

    assert isinstance(preprocess.convert_to_unicode(test_str), str)
    assert isinstance(preprocess.convert_to_unicode(test_byte), str)
