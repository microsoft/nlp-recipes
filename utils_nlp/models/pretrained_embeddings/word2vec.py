# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Functions to help users load and extract Word2Vec pretrained embeddings."""

import gzip
import os

from gensim.models.keyedvectors import KeyedVectors

from utils_nlp.dataset.url_utils import maybe_download
from utils_nlp.models.pretrained_embeddings import WORD2VEC_URL


def _extract_word2vec_vectors(zip_path, dest_filepath):
    """ Extracts word2vec embeddings from bin.gz archive

    Args:
        zip_path: Path to the downloaded compressed file.
        dest_filepath: Final destination file path to the extracted zip file.
    """

    if os.path.exists(zip_path):
        with gzip.GzipFile(zip_path, "rb") as f_in, open(
            dest_filepath, "wb"
        ) as f_out:
            f_out.writelines(f_in)
    else:
        raise Exception("Zipped file not found!")

    os.remove(zip_path)


def _download_word2vec_vectors(
    download_dir, file_name="GoogleNews-vectors-negative300.bin.gz"
):
    """ Downloads pretrained word vectors trained on GoogleNews corpus. You can
    directly download the vectors from here:
    https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz

    Args:
        download_dir (str): File path to download the file
        file_name (str) : File name given by default but can be changed by the user.

    Returns:
        str: file_path to the downloaded vectors.
    """

    return maybe_download(
        WORD2VEC_URL, filename=file_name, work_directory=download_dir
    )


def _maybe_download_and_extract(dest_path, file_name):
    """ Downloads and extracts Word2vec vectors if they don’t already exist

    Args:
        dest_path: Path to the directory where the vectors will be extracted.
        file_name: File name of the word2vec vector file.

    Returns:
         str: File path to the word2vec vector file.
    """

    dir_path = os.path.join(dest_path, "word2vec")
    file_path = os.path.join(dir_path, file_name)

    if not os.path.exists(file_path):
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        filepath = _download_word2vec_vectors(dir_path)
        _extract_word2vec_vectors(filepath, file_path)
    else:
        print("Vector file already exists. No changes made.")

    return file_path


def load_pretrained_vectors(
    dir_path, file_name="GoogleNews-vectors-negative300.bin", limit=None
):
    """ Method that loads word2vec vectors. Downloads if it doesn't exist.

    Args:
        file_name(str): Name of the word2vec file.
        dir_path(str): Path to the directory where word2vec vectors exist or will be
        downloaded.
        limit(int): Number of word vectors that is loaded from gensim. This option
        allows us to save RAM space and avoid memory errors.

    Returns:
        gensim.models.keyedvectors.Word2VecKeyedVectors: Loaded word2vectors

    """
    file_path = _maybe_download_and_extract(dir_path, file_name)
    word2vec_vectors = KeyedVectors.load_word2vec_format(
        file_path, binary=True, limit=limit
    )

    return word2vec_vectors
