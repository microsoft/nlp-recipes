# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import gzip
import os

from gensim.models.keyedvectors import KeyedVectors

from utils_nlp.dataset.url_utils import maybe_download


def extract_word2vec_corpus(zip_path, zip_dest_file_path):
    """ Extracts word2vec embeddings from bin.gz archive

    Args:
        zip_path: Path to the downloaded compressed file.
        zip_dest_file_path: Final destination file path to the extracted zip file.

    Returns:
        Returns the absolute path to the extracted folder.
    """

    if os.path.exists(zip_path):
        with gzip.GzipFile(zip_path, "rb") as f_in, open(
            zip_dest_file_path, "wb"
        ) as f_out:
            f_out.writelines(f_in)
    else:
        raise Exception("Zipped file not found!")

    os.remove(zip_path)


def download_word2vec_corpus(
    download_dir, file_name="GoogleNews-vectors-negative300.bin.gz"
):
    """ Downloads pretrained word vectors trained on GoogleNews corpus. You can
    directly download the vectors from here:
    https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz

    Args:
        download_dir (str): File path to download the file
        file_name (str) : File name given by default but can be changed by the user.

    Returns:
        file_path to the downloaded vectors.
    """

    url = (
        "https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300"
        ".bin.gz "
    )
    return maybe_download(url, filename=file_name, work_directory=download_dir)


def _maybe_download_and_extract(dest_path, file_name):
    """ Downloads and extracts Word2vec vectors if they don’t already exist

    Args:
        dest_path: Final path where the vectors will be extracted.

    """

    word2vec_dir_path = os.path.join(dest_path, "word2vec")
    word2vec_file_path = os.path.join(word2vec_dir_path, file_name)

    if not os.path.exists(word2vec_file_path):
        if not os.path.exists(word2vec_dir_path):
            os.makedirs(word2vec_dir_path)
        filepath = download_word2vec_corpus(word2vec_dir_path)
        extract_word2vec_corpus(filepath, word2vec_file_path)
    else:
        print("Vector file already exists. No changes made.")

    return word2vec_file_path


def load_pretrained_vectors(
    dir_path, file_name="GoogleNews-vectors-negative300.bin"
):
    """ Method that loads word2vec vectors. Downloads if it doesn't exist.

    Args:
        file_name(str): Name of the word2vec file.
        dir_path(str): Path to the directory where word2vec vectors exist or will be
        downloaded.

    Returns: Loaded word2vectors (gensim.models.keyedvectors.Word2VecKeyedVectors)

    """
    file_path = _maybe_download_and_extract(dir_path, file_name)
    word2vec_vectors = KeyedVectors.load_word2vec_format(
        file_path, binary=True
    )
    print(type(word2vec_vectors))
    return word2vec_vectors
