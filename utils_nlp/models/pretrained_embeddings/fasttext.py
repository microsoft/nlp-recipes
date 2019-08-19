# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Functions to help users load and extract fastText pretrained embeddings."""

import os
import zipfile

from gensim.models.fasttext import load_facebook_model

from utils_nlp.dataset.url_utils import maybe_download
from utils_nlp.models.pretrained_embeddings import FASTTEXT_EN_URL


def _extract_fasttext_vectors(zip_path, dest_path="."):
    """ Extracts fastText embeddings from zip file.

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


def _download_fasttext_vectors(download_dir, file_name="wiki.simple.zip"):
    """ Downloads pre-trained word vectors for English, trained on Wikipedia using
    fastText. You can directly download the vectors from here:
    https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.simple.zip

    For the full version of pre-trained word vectors, change the url for
    FASTTEXT_EN_URL to https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.en.zip
    in __init__.py

    Args:
        download_dir (str): File path to download the file
        file_name (str) : File name given by default but can be changed by the user.

    Returns:
        str: file_path to the downloaded vectors.
    """

    return maybe_download(
        FASTTEXT_EN_URL, filename=file_name, work_directory=download_dir
    )


def _maybe_download_and_extract(dest_path, file_name):
    """ Downloads and extracts fastText vectors if they don’t already exist

    Args:
        dest_path(str): Final path where the vectors will be extracted.
        file_name(str): File name of the fastText vector file.

    Returns:
        str: File path to the fastText vector file.
    """

    dir_path = os.path.join(dest_path, "fastText")
    file_path = os.path.join(dir_path, file_name)

    if not os.path.exists(file_path):
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        zip_path = _download_fasttext_vectors(dir_path)
        _extract_fasttext_vectors(zip_path, dir_path)
    else:
        print("Vector file already exists. No changes made.")

    return file_path


def load_pretrained_vectors(dest_path, file_name="wiki.simple.bin"):
    """ Method that loads fastText vectors. Downloads if it doesn't exist.

    Args:
        file_name(str): Name of the fastText file.
        dest_path(str): Path to the directory where fastText vectors exist or will be
        downloaded.

    Returns:
        gensim.models.fasttext.load_facebook_model: Loaded word2vectors

    """

    file_path = _maybe_download_and_extract(dest_path, file_name)
    model = load_facebook_model(file_path)
    return model
