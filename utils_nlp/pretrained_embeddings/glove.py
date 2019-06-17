# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import logging
import os
import zipfile

from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.test.utils import get_tmpfile

from . import GLOVE_URL, VectorLoader

module_logger = logging.getLogger(__name__)

class GloVeVectorLoader(VectorLoader):
    name = "gloVe"

    @classmethod
    def download_vectors(cls, download_dir, file_name="glove.840B.300d.zip"):
        """ Downloads gloVe word vectors trained on Common Crawl corpus. You can
        directly download the vectors from here:
        http://nlp.stanford.edu/data/glove.840B.300d.zip

        Args:
            download_dir (str): File path to download the file
            file_name (str) : File name given by default but can be changed by the user.

        Returns:
            str: file_path to the downloaded vectors.
        """

        return self.maybe_download(GLOVE_URL, file_name=file_name, download_dir=download_dir)


def load_pretrained_vectors(dir_path, file_name="glove.840B.300d.txt", limit=None):
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

    file_path = GloVeVectorLoader._maybe_download_and_extract(
        dir_path, file_name)
    tmp_file = get_tmpfile("test_word2vec.txt")
    glove2word2vec(file_path, tmp_file)
    return KeyedVectors.load_word2vec_format(tmp_file, limit=limit)
