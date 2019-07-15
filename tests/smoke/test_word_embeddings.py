# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os

import pytest
from gensim.models.fasttext import FastText
from gensim.models.keyedvectors import Word2VecKeyedVectors

from utils_nlp.models.pretrained_embeddings.fasttext import (
    load_pretrained_vectors as load_fasttext,
)
from utils_nlp.models.pretrained_embeddings.glove import (
    load_pretrained_vectors as load_glove,
)
from utils_nlp.models.pretrained_embeddings.word2vec import (
    load_pretrained_vectors as load_word2vec,
)


@pytest.mark.smoke
def test_load_pretrained_vectors_word2vec(tmp_path):
    filename = "GoogleNews-vectors-negative300.bin"
    model = load_word2vec(tmp_path, limit=500000)
    filepath = os.path.join(os.path.join(tmp_path, "word2vec"), filename)
    statinfo = os.stat(filepath)
    assert statinfo.st_size == 3644258522
    assert isinstance(model, Word2VecKeyedVectors)
    assert len(model.vocab) == 500000


@pytest.mark.smoke
def test_load_pretrained_vectors_glove(tmp_path):
    filename = "glove.840B.300d.txt"
    model = load_glove(tmp_path, limit=50000)
    filepath = os.path.join(os.path.join(tmp_path, "gloVe"), filename)
    statinfo = os.stat(filepath)
    assert statinfo.st_size == 5646236541
    assert isinstance(model, Word2VecKeyedVectors)
    assert len(model.vocab) == 50000


@pytest.mark.smoke
def test_load_pretrained_vectors_fasttext(tmp_path):
    filename = "wiki.simple.bin"
    model = load_fasttext(tmp_path)
    filepath = os.path.join(os.path.join(tmp_path, "fastText"), filename)
    statinfo = os.stat(filepath)
    assert statinfo.st_size == 2668450750
    assert isinstance(model, FastText)


