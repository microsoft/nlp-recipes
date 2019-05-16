# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import os
import shutil
from pathlib import Path

from gensim.models.fasttext import FastText
from gensim.models.keyedvectors import Word2VecKeyedVectors

from utils_nlp.pretrained_embeddings.fasttext import (
    load_pretrained_vectors as load_fasttext,
)
from utils_nlp.pretrained_embeddings.glove import (
    load_pretrained_vectors as load_glove,
)
from utils_nlp.pretrained_embeddings.word2vec import (
    load_pretrained_vectors as load_word2vec,
)


def test_load_pretrained_vectors_word2vec():
    dir_path = "temp_data/"
    file_path = os.path.join(
        os.path.join(dir_path, "word2vec"),
        "GoogleNews-vectors-negative300.bin",
    )

    assert isinstance(load_word2vec(dir_path), Word2VecKeyedVectors)

    model = load_word2vec(dir_path, limit=500000)
    assert isinstance(model, Word2VecKeyedVectors)
    assert (len(model.wv.vocab) == 500000)

    file_path = Path(file_path)
    assert file_path.is_file()

    shutil.rmtree(os.path.join(os.getcwd(), dir_path))


def test_load_pretrained_vectors_glove():
    dir_path = "temp_data/"
    file_path = os.path.join(
        os.path.join(dir_path, "gloVe"), "glove.840B.300d.txt"
    )

    assert isinstance(load_glove(dir_path), Word2VecKeyedVectors)

    model = load_glove(dir_path, limit=50000)
    assert isinstance(model, Word2VecKeyedVectors)
    assert (len(model.wv.vocab) == 50000)

    file_path = Path(file_path)
    assert file_path.is_file()

    shutil.rmtree(os.path.join(os.getcwd(), dir_path))


def test_load_pretrained_vectors_fasttext():
    dir_path = "temp_data/"
    file_path = os.path.join(os.path.join(dir_path, "fastText"), "wiki.en.bin")

    assert isinstance(load_fasttext(dir_path), FastText)

    file_path = Path(file_path)
    assert file_path.is_file()

    shutil.rmtree(os.path.join(os.getcwd(), dir_path))
