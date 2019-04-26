# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import gzip
import os

from gensim.models.keyedvectors import KeyedVectors

from utils_nlp.dataset.url_utils import maybe_download


def load_word2vec():
    # Todo : Move to azure blob and get rid of this path.
    file_path = (
        "../../../Pretrained Vectors/GoogleNews-vectors-negative300.bin"
    )
    model = KeyedVectors.load_word2vec_format(file_path, binary=True)
    print(type(model))


def extract_word2vec_corpus(
    zip_path,
    zip_dest_path=".",
    zipped_file_name="GoogleNews-vectors-negative300.bin",
):
    """ Extracts word2vec embeddings from bin.gz archive

    Args:
        zipped_file_name: File name for the extracted file.
        zip_path: Path to the downloaded compressed file.
        zip_dest_path: Destination path to the extracted zip file. This will be the
        current folder by default.

    Returns:
        Returns the absolute path to the extracted folder.
    """

    assert os.path.exists(zip_path) and os.path.exists(zip_dest_path)
    zipped_file_path = os.path.join(zip_dest_path, zipped_file_name)

    with open(zip_path, "rb") as f_in, gzip.open(
        zipped_file_path, "wb"
    ) as f_out:
        f_out.writelines(f_in)

    os.remove(zip_path)
    return zipped_file_path


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


def _maybe_download_and_extract(dest_path):
    """ Downloads and extracts Word2vec vectors if they don’t already exist

    Args:
        dest_path: Final path where the vectors will be extracted.

    """

    word2vec_path = os.path.join(dest_path, "word2vec")
    extracted_path = None

    if not os.path.exists(word2vec_path):
        os.makedirs(word2vec_path)
        filepath = download_word2vec_corpus(word2vec_path)
        extracted_path = extract_word2vec_corpus(filepath, word2vec_path)
    else:
        print("Word2vec already exists. No changes made.")

    return extracted_path


if __name__ == "__main__":
    print(_maybe_download_and_extract(r"C:\Projects\NLP-BP\NLP\data"))
