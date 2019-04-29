# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import os
import shutil
import pytest
from pathlib import Path

from gensim.models.keyedvectors import Word2VecKeyedVectors

from utils_nlp.pretrained_embeddings.word2vec import (
    load_pretrained_vectors,
    download_word2vec_corpus,
    extract_word2vec_corpus
)


def test_load_pretrained_vectors_word2vec():
    dir_path = "temp_data/"
    assert isinstance(load_pretrained_vectors(dir_path), Word2VecKeyedVectors)
    shutil.rmtree(os.path.join(os.getcwd(), dir_path))


def test_download_word2vec_corpus():
    dir_path = "temp_data/"
    os.makedirs(dir_path)
    file_path = Path(download_word2vec_corpus(dir_path))
    assert file_path.is_file()
    shutil.rmtree(os.path.join(os.getcwd(), dir_path))


def test_extract_word2vec_corpus():
    dir_path = "temp_data/"
    os.makedirs(dir_path)
    file_name = "GoogleNews-vectors-negative300.bin"
    word2vec_file_path = os.path.join(dir_path, file_name)

    with pytest.raises(Exception) as e_info:
        extract_word2vec_corpus(dir_path, word2vec_file_path)

    zip_path = download_word2vec_corpus(dir_path)
    extract_word2vec_corpus(zip_path, word2vec_file_path)

    file_path = Path(word2vec_file_path)
    assert file_path.is_file()

    shutil.rmtree(os.path.join(os.getcwd(), dir_path))

