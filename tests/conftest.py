# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# NOTE: This file is used by pytest to inject fixtures automatically.
# As it is explained in the documentation
# https://docs.pytest.org/en/latest/fixture.html:
# "If during implementing your tests you realize that you want to use
# a fixture function from multiple test files you can move it to a conftest.py
# file. You don’t need to import the fixture you want to use in a test, it
# automatically gets discovered by pytest."

import os
from tempfile import TemporaryDirectory

import pytest
from tests.notebooks_common import path_notebooks

from utils_nlp.bert.common import Language
from utils_nlp.bert.common import Tokenizer as BERTTokenizer


@pytest.fixture(scope="module")
def notebooks():
    folder_notebooks = path_notebooks()

    # Path for the notebooks
    paths = {
        "similarity_embeddings_baseline": os.path.join(
            folder_notebooks, "sentence_similarity", "baseline_deep_dive.ipynb"
        ),
        "senteval_local": os.path.join(
            folder_notebooks, "sentence_similarity", "senteval_local.ipynb"
        ),
        "senteval_azureml": os.path.join(
            folder_notebooks, "sentence_similarity", "senteval_azureml.ipynb"
        ),
        "embedding_trainer": os.path.join(
            folder_notebooks, "embeddings", "embedding_trainer.ipynb"
        ),
    }
    return paths


@pytest.fixture
def tmp(tmp_path_factory):
    td = TemporaryDirectory(dir=tmp_path_factory.getbasetemp())
    try:
        yield td.name
    finally:
        td.cleanup()


@pytest.fixture(scope="module")
def ner_test_data():
    UNIQUE_LABELS = ["O", "I-LOC", "I-MISC", "I-PER", "I-ORG", "X"]
    LABEL_MAP = {label: i for i, label in enumerate(UNIQUE_LABELS)}
    TRAILING_TOKEN_MASK = [[True] * 20]
    false_pos = [1, 2]
    for p in false_pos:
        TRAILING_TOKEN_MASK[0][p] = False
    INPUT_LABEL_IDS = [
        [3, 5, 5, 0, 0, 0, 0, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ]
    return {
        "INPUT_TEXT": [
            [
                "Johnathan",
                "is",
                "studying",
                "in",
                "the",
                "University",
                "of",
                "Michigan",
                ".",
            ]
        ],
        "INPUT_TEXT_SINGLE": [
            "Johnathan",
            "is",
            "studying",
            "in",
            "the",
            "University",
            "of",
            "Michigan",
            ".",
        ],
        "INPUT_LABELS": [
            ["I-PER", "O", "O", "O", "O", "I-ORG", "I-ORG", "I-ORG", "O"]
        ],
        "INPUT_LABELS_SINGLE": [
            "I-PER",
            "O",
            "O",
            "O",
            "O",
            "I-ORG",
            "I-ORG",
            "I-ORG",
            "O",
        ],
        "INPUT_LABELS_WRONG": [
            ["I-PER", "O", "O", "O", "O", "I-ORG", "I-ORG", "I-ORG"]
        ],
        "INPUT_TOKEN_IDS": [
            [
                1287,
                9779,
                1389,
                1110,
                5076,
                1107,
                1103,
                1239,
                1104,
                3312,
                119,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ]
        ],
        "INPUT_LABEL_IDS": INPUT_LABEL_IDS,
        "INPUT_MASK": [[1] * 11 + [0] * 9],
        "PREDICTED_LABELS": [
            [3, 5, 5, 0, 0, 0, 0, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        ],
        "TRAILING_TOKEN_MASK": TRAILING_TOKEN_MASK,
        "UNIQUE_LABELS": UNIQUE_LABELS,
        "LABEL_MAP": LABEL_MAP,
        "EXPECTED_TOKENS_NO_PADDING": [
            [
                "I-PER",
                "X",
                "X",
                "O",
                "O",
                "O",
                "O",
                "I-ORG",
                "I-ORG",
                "I-ORG",
                "O",
            ]
        ],
        "EXPECTED_TOKENS_NO_PADDING_NO_TRAILING": [
            ["I-PER", "O", "O", "O", "O", "I-ORG", "I-ORG", "I-ORG", "O"]
        ],
        "EXPECTED_TRAILING_TOKEN_MASK": TRAILING_TOKEN_MASK,
        "EXPECTED_LABEL_IDS": INPUT_LABEL_IDS,
    }


@pytest.fixture()
def bert_english_tokenizer():
    return BERTTokenizer(language=Language.ENGLISHCASED, to_lower=False)

@pytest.fixture(scope="module")
def baseline_results():
    return {
        "Word2vec Cosine": 0.6337760059182685,
        "Word2vec Cosine with Stop Words": 0.647674307797345,
        "Word2vec WMD": 0.6578256301323717,
        "Word2vec WMD with Stop Words": 0.5697910628727217,
        "GLoVe Cosine": 0.642064729899729,
        "GLoVe Cosine with Stop Words": 0.5639670964748242,
        "GLoVe WMD": 0.6272339050920003,
        "GLoVe WMD with Stop Words": 0.48560149551724,
        "fastText Cosine": 0.6288780924569853,
        "fastText Cosine with Stop Words": 0.5958470751204786,
        "fastText WMD": 0.527520845792085,
        "fastText WMD with Stop Words": 0.44198752510004097,
        "TF-IDF Cosine": 0.6683811410442562,
        "TF-IDF Cosine with Stop Words": 0.7034695168223282,
        "Doc2vec Cosine": 0.5082738341805563,
        "Doc2vec Cosine with Stop Words": 0.4116873013460912,
    }


@pytest.fixture(scope="module")
def gensen_senteval_results():
    return {
        "pearson": {
            "STSBenchmark": 0.778,
            "STS12": 0.598,
            "STS13": 0.542,
            "STS14": 0.639,
            "STS15": 0.727,
            "STS16": 0.659,
        },
        "spearman": {
            "STSBenchmark": 0.780,
            "STS12": 0.603,
            "STS13": 0.556,
            "STS14": 0.625,
            "STS15": 0.729,
            "STS16": 0.663,
        },
    }
