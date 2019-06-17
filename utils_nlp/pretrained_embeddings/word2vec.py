# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import gzip
import logging
import os

from gensim.models.keyedvectors import KeyedVectors

from . import WORD2VEC_URL

module_logger = logging.getLogger(__name__)


class Word2vecVectorLoader(VectorLoader):
    name = "word2vec"

    @classmethod
    def download_vectors(cls, download_dir,
                         file_name="GoogleNews-vectors-negative300.bin.gz"):
        """ Downloads pretrained word vectors trained on GoogleNews corpus. You can
        directly download the vectors from here:
        https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz

        Args:
            download_dir (str): File path to download the file
            file_name (str) : File name given by default but can be changed by the user.

        Returns:
            str: file_path to the downloaded vectors.
        """

        return self.maybe_download(
                WORD2VEC_URL, file_name=file_name, download_dir=download_dir)


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
    file_path = Word2vecVectorLoader._maybe_download_and_extract(dir_path,
            file_name)
    return KeyedVectors.load_word2vec_format(file_path, binary=True, limit=limit)
