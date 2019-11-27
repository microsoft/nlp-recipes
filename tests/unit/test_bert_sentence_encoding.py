# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pytest

from utils_nlp.models.bert.common import Language
from utils_nlp.models.bert.sequence_encoding import BERTSentenceEncoder, PoolingStrategy
from sklearn.metrics.pairwise import cosine_similarity


@pytest.fixture()
def data():
    return [
        "how old are you?",
        "what's your age?",
        "my phone is good",
        "your cellphone looks great.",
    ]


def test_sentence_encoding(tmp, data):
    se = BERTSentenceEncoder(
        language=Language.ENGLISH,
        num_gpus=0,
        to_lower=True,
        max_len=128,
        layer_index=-2,
        pooling_strategy=PoolingStrategy.MEAN,
        cache_dir=tmp,
    )

    result = se.encode(data, as_numpy=False)
    similarity = cosine_similarity(result["values"].values.tolist())
    assert similarity[0, 0] > similarity[1, 0]
    assert similarity[0, 1] > similarity[0, 2]
