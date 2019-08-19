# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Functions to help users load and extract GloVe pretrained embeddings."""

import os
import zipfile

from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.test.utils import get_tmpfile

from utils_nlp.dataset.url_utils import maybe_download
from utils_nlp.models.pretrained_embeddings import GLOVE_URL


def _extract_glove_vectors(zip_path, dest_path="."):
    """ Extracts gloVe embeddings from zip file.

    Args:
        zip_path(str): Path to the downloaded compressed zip file.
        dest_path(str): Final destination directory path to the extracted zip file.
        Picks the current working directory by default.

    Returns:
        str: Returns the absolute path to the extracted folder.
    """

    if os.path.exists(zip_path):
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(path=dest_path)
    else:
        raise Exception("Zipped file not found!")

    os.remove(zip_path)
    return dest_path


def _download_glove_vectors(download_dir, file_name="glove.840B.300d.zip"):
    """ Downloads gloVe word vectors trained on Common Crawl corpus. You can
    directly download the vectors from here:
    http://nlp.stanford.edu/data/glove.840B.300d.zip

    Args:
        download_dir (str): File path to download the file
        file_name (str) : File name given by default but can be changed by the user.

    Returns:
        str: file_path to the downloaded vectors.
    """

    return maybe_download(
        GLOVE_URL, filename=file_name, work_directory=download_dir
    )


def _maybe_download_and_extract(dest_path, file_name):
    """ Downloads and extracts gloVe vectors if they don’t already exist

    Args:
        dest_path(str): Final path where the vectors will be extracted.
        file_name(str): File name of the gloVe vector file.

    Returns:
        str: File path to the gloVe vector file.
    """

    dir_path = os.path.join(dest_path, "gloVe")
    file_path = os.path.join(dir_path, file_name)

    if not os.path.exists(file_path):
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        filepath = _download_glove_vectors(dir_path)
        _extract_glove_vectors(filepath, dir_path)
    else:
        print("Vector file already exists. No changes made.")

    return file_path


def download_and_extract(dir_path, file_name="glove.840B.300d.txt"):
    """ Downloads and extracts gloVe vectors if they don’t already exist

    Args:
        dir_path(str): Final path where the vectors will be extracted.
        file_name(str): File name of the gloVe vector file.

    Returns:
        str: File path to the gloVe vector file.
    """

    return _maybe_download_and_extract(dir_path, file_name)


def load_pretrained_vectors(
    dir_path, file_name="glove.840B.300d.txt", limit=None
):
    """ Method that loads gloVe vectors. Downloads if it doesn't exist.

    Args:
        file_name(str): Name of the gloVe file.
        dir_path(str): Path to the directory where gloVe vectors exist or will be
        downloaded.
        limit(int): Number of word vectors that is loaded from gensim. This option
        allows us to save RAM space and avoid memory errors.

    Returns:
        gensim.models.keyedvectors.Word2VecKeyedVectors: Loaded word2vectors
    """

    file_path = _maybe_download_and_extract(dir_path, file_name)
    tmp_file = get_tmpfile("test_word2vec.txt")

    # Convert GloVe format to word2vec
    _ = glove2word2vec(file_path, tmp_file)

    model = KeyedVectors.load_word2vec_format(tmp_file, limit=limit)
    os.remove(tmp_file)

    return model
