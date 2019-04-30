# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import os
import shutil
from pathlib import Path

from gensim.models.keyedvectors import Word2VecKeyedVectors

from utils_nlp.pretrained_embeddings.glove import (
    load_pretrained_vectors as load_glove,
)
from utils_nlp.pretrained_embeddings.word2vec import (
    load_pretrained_vectors as load_word2vec,
)


def test_load_pretrained_vectors_word2vec():
    dir_path = "temp_data/"
    file_name = "GoogleNews-vectors-negative300.bin"
    word2vec_file_path = os.path.join(
        os.path.join(dir_path, "word2vec"), file_name
    )

    assert isinstance(load_word2vec(dir_path), Word2VecKeyedVectors)

    file_path = Path(word2vec_file_path)
    assert file_path.is_file()

    shutil.rmtree(os.path.join(os.getcwd(), dir_path))


def test_load_pretrained_vectors_glove():
    dir_path = "temp_data/"
    file_name = "glove.840B.300d.txt"
    glove_file_path = os.path.join(os.path.join(dir_path, "gloVe"), file_name)

    assert isinstance(load_glove(dir_path), Word2VecKeyedVectors)

    file_path = Path(glove_file_path)
    assert file_path.is_file()

    shutil.rmtree(os.path.join(os.getcwd(), dir_path))
