# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
from urllib.request import urlretrieve
import tarfile
from zipfile import ZipFile
import logging
from contextlib import contextmanager
from tempfile import TemporaryDirectory
from tqdm import tqdm


log = logging.getLogger(__name__)


class TqdmUpTo(tqdm):
    """Wrapper class for the progress bar tqdm to get `update_to(n)` functionality"""

    def update_to(self, b=1, bsize=1, tsize=None):
        """A progress bar showing how much is left to finish the opperation

        Args:
            b (int): Number of blocks transferred so far.
            bsize (int): Size of each block (in tqdm units).
            tsize (int): Total size (in tqdm units). 
        """
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)  # will also set self.n = b * bsize


def maybe_download(
    url, filename=None, work_directory=".", expected_bytes=None
):
    """Download a file if it is not already downloaded.

    Args:
        filename (str): File name.
        work_directory (str): Working directory.
        url (str): URL of the file to download.
        expected_bytes (int): Expected file size in bytes.

    Returns:
        str: File path of the file downloaded.
    """
    if filename is None:
        filename = url.split("/")[-1]
    filepath = os.path.join(work_directory, filename)
    if not os.path.exists(filepath):
        with TqdmUpTo(unit="B", unit_scale=True) as t:
            filepath, _ = urlretrieve(url, filepath, reporthook=t.update_to)
    else:
        log.debug("File {} already downloaded".format(filepath))
    if expected_bytes is not None:
        statinfo = os.stat(filepath)
        if statinfo.st_size != expected_bytes:
            os.remove(filepath)
            raise IOError("Failed to verify {}".format(filepath))

    return filepath


def extract_tar(file_path, dest_path="."):
    """Extracts all contents of a tar archive file.
    Args:
        file_path (str): Path of file to extract.
        dest_path (str, optional): Destination directory. Defaults to ".".
    """
    if not os.path.exists(file_path):
        raise IOError("File doesn't exist")
    if not os.path.exists(dest_path):
        raise IOError("Destination directory doesn't exist")
    with tarfile.open(file_path) as t:
        t.extractall(path=dest_path)


def extract_zip(file_path, dest_path="."):
    """Extracts all contents of a zip archive file.
    Args:
        file_path (str): Path of file to extract.
        dest_path (str, optional): Destination directory. Defaults to ".".
    """
    if not os.path.exists(file_path):
        raise IOError("File doesn't exist")
    if not os.path.exists(dest_path):
        raise IOError("Destination directory doesn't exist")
    with ZipFile(file_path) as z:
        z.extractall(path=dest_path)


@contextmanager
def download_path(path):
    tmp_dir = TemporaryDirectory()
    if path is None:
        path = tmp_dir.name
    else:
        path = os.path.realpath(path)

    try:
        yield path
    finally:
        tmp_dir.cleanup()
