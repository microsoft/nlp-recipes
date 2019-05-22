# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os

import pytest
import pandas as pd
import numpy as np

import utils_nlp.dataset.preprocess as preprocess


@pytest.fixture(scope="module")
def get_sentences():
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

    df = pd.DataFrame(sentences)
    df.columns = ["s1", "s2", "s3", "s4", "s5"]
    return df


def test_to_lowercase_all(get_sentences):
    df = get_sentences
    ldf = preprocess.to_lowercase_all(df)
    assert sum(map(lambda x: x.islower(), ldf.values.flatten())) == len(
        ldf.values.flatten()
    )


def test_to_lowercase_subset(get_sentences):
    df = get_sentences
    ldf = preprocess.to_lowercase(df, column_names=["s4"])
    assert sum(map(lambda x: x.islower(), ldf.s4.values.flatten())) == len(
        ldf.s4.values.flatten()
    )


def test_to_spacy_tokens(get_sentences):
    df = get_sentences
    sentence_cols = ["s1", "s2"]
    token_cols = ["t1", "t2"]
    token_df = preprocess.to_spacy_tokens(
        df, sentence_cols=sentence_cols, token_cols=token_cols
    )
    assert token_df.shape[1] == df.shape[1] + len(token_cols) and sum(
        list(
            map(lambda x: (token_df[x].apply(type) == list).all(), token_cols)
        )
    ) == len(token_cols)


def test_rm_spacy_stopwords(get_sentences):
    df = get_sentences
    sentence_cols = ["s1", "s2"]
    stop_cols = ["stop1", "stop2"]
    stop_df = preprocess.rm_spacy_stopwords(
        df, sentence_cols=sentence_cols, stop_cols=stop_cols
    )
    assert stop_df.shape[1] == df.shape[1] + len(stop_cols) and sum(
        list(map(lambda x: (stop_df[x].apply(type) == list).all(), stop_cols))
    ) == len(stop_cols)


def test_to_nltk_tokens(get_sentences):
    df = get_sentences
    sentence_cols = ["s1", "s2"]
    token_cols = ["t1", "t2"]
    token_df = preprocess.to_nltk_tokens(
        df, sentence_cols=sentence_cols, token_cols=token_cols
    )
    assert token_df.shape[1] == df.shape[1] + len(token_cols) and sum(
        list(
            map(lambda x: (token_df[x].apply(type) == list).all(), token_cols)
        )
    ) == len(token_cols)


def test_rm_nltk_stopwords(get_sentences):
    df = get_sentences
    sentence_cols = ["s1", "s2"]
    stop_cols = ["stop1", "stop2"]
    stop_df = preprocess.rm_nltk_stopwords(
        df, sentence_cols=sentence_cols, stop_cols=stop_cols
    )
    assert stop_df.shape[1] == df.shape[1] + len(stop_cols) and sum(
        list(map(lambda x: (stop_df[x].apply(type) == list).all(), stop_cols))
    ) == len(stop_cols)
