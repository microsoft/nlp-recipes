# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import zipfile

from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.test.utils import get_tmpfile

from utils_nlp.dataset.url_utils import maybe_download


def _extract_glove_vectors(zip_path, zip_dest_dir="."):
    """ Extracts gloVe embeddings from zip file.

    Args:
        zip_path(str): Path to the downloaded compressed zip file.
        zip_dest_dir(str): Final destination directory path to the extracted zip file.
        Picks the current working directory by default.

    Returns:
        Returns the absolute path to the extracted folder.
    """

    if os.path.exists(zip_path):
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(path=zip_dest_dir)
    else:
        raise Exception("Zipped file not found!")

    os.remove(zip_path)
    return zip_dest_dir


def _download_glove_vectors(download_dir, file_name="glove.840B.300d.zip"):
    """ Downloads gloVe word vectors trained on Common Crawl corpus. You can
    directly download the vectors from here:
    http://nlp.stanford.edu/data/glove.840B.300d.zip

    Args:
        download_dir (str): File path to download the file
        file_name (str) : File name given by default but can be changed by the user.

    Returns:
        file_path to the downloaded vectors.
    """

    url = "http://nlp.stanford.edu/data/glove.840B.300d.zip"
    return maybe_download(url, filename=file_name, work_directory=download_dir)


def _maybe_download_and_extract(dest_path, file_name):
    """ Downloads and extracts gloVe vectors if they don’t already exist

    Args:
        dest_path(str): Final path where the vectors will be extracted.
        file_name(str): File name of the gloVe vector file.

    Returns: File path to the gloVe vector file.
    """

    glove_dir_path = os.path.join(dest_path, "gloVe")
    glove_file_path = os.path.join(glove_dir_path, file_name)

    if not os.path.exists(glove_file_path):
        if not os.path.exists(glove_dir_path):
            os.makedirs(glove_dir_path)
        filepath = _download_glove_vectors(glove_dir_path)
        _extract_glove_vectors(filepath, glove_dir_path)
    else:
        print("Vector file already exists. No changes made.")

    return glove_file_path


def load_pretrained_vectors(dir_path, file_name="glove.840B.300d.txt"):
    """ Method that loads gloVe vectors. Downloads if it doesn't exist.

    Args:
        file_name(str): Name of the gloVe file.
        dir_path(str): Path to the directory where gloVe vectors exist or will be
        downloaded.

    Returns: Loaded word2vectors (gensim.models.keyedvectors.Word2VecKeyedVectors)

    """

    file_path = _maybe_download_and_extract(dir_path, file_name)
    tmp_file = get_tmpfile("test_word2vec.txt")
    _ = glove2word2vec(file_path, tmp_file)
    model = KeyedVectors.load_word2vec_format(tmp_file)

    return model
