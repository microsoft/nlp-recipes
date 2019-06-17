# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import logging

from utils_nlp.dataset.url_utils import maybe_download

WORD2VEC_URL = 'https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz'
FASTTEXT_EN_URL = 'https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.simple.zip'
GLOVE_URL = 'http://nlp.stanford.edu/data/glove.840B.300d.zip'

module_logger = logging.getLogger(__name__)


class VectorLoader(object):
    @classmethod
    def extract_vectors(cls, zip_path, dest_filepath):
        """ Extracts word2vec embeddings from bin.gz archive

        Args:
            zip_path: Path to the downloaded compressed file.
            dest_filepath: Final destination file path to the extracted zip file.
        """

        try:
            if os.path.exists(zip_path):
                with gzip.GzipFile(zip_path, "rb") as f_in, open(
                    dest_filepath, "wb"
                ) as f_out:
                    f_out.writelines(f_in)
            else:
                raise Exception("Zipped file not found! For zip_path {}".format(zip_path))

        finally:
            if os.path.exists(zip_path):
                os.remove(zip_path)

    @classmethod
    def _maybe_download_and_extract(cls, dest_path, file_name):
        """ Downloads and extracts vectors if they don’t already exist

        Args:
            dest_path(str): Final path where the vectors will be extracted.
            file_name(str): File name of the vector file.

        Returns:
            str: File path to the vector file.
        """

        dir_path = os.path.join(dest_path, cls.name)
        file_path = os.path.join(dir_path, file_name)

        if not os.path.exists(file_path):
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            zip_path = cls.download_vectors(dir_path)
            cls.extract_vectors(zip_path, dir_path)
        else:
            module_logger.info("Vector file already exists. No changes made.")

        return file_path

    @staticmethod
    def download_vectors(cls, *args, **kwargs):
        raise NotImplementedError()

    @staticmethod
    def extract_vectors(cls, *args, **kwargs):
        raise NotImplementedError()

    @staticmethod
    def maybe_download(url, file_name=None, download_dir=None)
        return maybe_download(url, filename=file_name, work_directory=download_dir)
