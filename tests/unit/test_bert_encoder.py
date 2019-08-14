# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pytest

from utils_nlp.models.bert.common import Language
from utils_nlp.models.bert.sequence_encoding import BERTSentenceEncoder

@pytest.fixture()
def data():
    return ["The quick brown fox jumps over the lazy dog", "the coffee is very acidic"]

def test_encoder(tmp, data):
    se = BERTSentenceEncoder(
        language=Language.ENGLISH,
        num_gpus=0,
        cache_dir=tmp,
    )
    embeddings = se.encode(data, as_numpy=True)
    assert len(embeddings) == 2
    assert len(embeddings[0]) == 768