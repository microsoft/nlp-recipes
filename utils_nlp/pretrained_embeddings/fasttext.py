# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import zipfile

from gensim.models.fasttext import load_facebook_model

from . import FASTTEXT_EN_URL, VectorLoader


class FastTextVectorLoader(VectorLoader):
    name = "fastText"

    def download_vectors(self, download_dir, file_name="wiki.simple.zip"):
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

        return self.maybe_download(
                FASTTEXT_EN_URL, file_name=file_name, download_dir=download_dir
    )



def load_pretrained_vectors(dest_path, file_name="wiki.simple.bin"):
    """ Method that loads fastText vectors. Downloads if it doesn't exist.

    Args:
        file_name(str): Name of the fastText file.
        dest_path(str): Path to the directory where fastText vectors exist or will be
        downloaded.

    Returns:
        gensim.models.fasttext.load_facebook_model: Loaded word2vectors

    """

    file_path = FastTextVectorLoader._maybe_download_and_extract(dest_path,
file_name)
    return load_facebook_model(file_path)
