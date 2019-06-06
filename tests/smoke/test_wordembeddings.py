# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import pytest

from utils_nlp.pretrained_embeddings.word2vec import (
    load_pretrained_vectors as load_word2vec,
)
from utils_nlp.pretrained_embeddings.glove import (
    load_pretrained_vectors as load_glove,
)
from utils_nlp.pretrained_embeddings.fasttext import (
    load_pretrained_vectors as load_fasttext,
)


@pytest.mark.smoke
def test_load_pretrained_vectors_word2vec(tmp_path):
    filename = "word2vec.bin"
    load_word2vec(tmp_path, filename)
    filepath = os.path.join(os.path.join(tmp_path, "word2vec"), filename)
    statinfo = os.stat(filepath)
    assert statinfo.st_size == 3644258522


@pytest.mark.smoke
def test_load_pretrained_vectors_glove(tmp_path):
    filename = "glove.840B.300d.txt"
    load_glove(tmp_path)
    filepath = os.path.join(os.path.join(tmp_path, "gloVe"), filename)
    statinfo = os.stat(filepath)
    assert statinfo.st_size == 5646236541


@pytest.mark.smoke
def test_load_pretrained_vectors_fasttext(tmp_path):
    filename = "wiki.simple.bin"
    load_fasttext(tmp_path)
    filepath = os.path.join(os.path.join(tmp_path, "fastText"), filename)
    statinfo = os.stat(filepath)
    assert statinfo.st_size == 2668450750
