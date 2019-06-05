# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import requests
import logging
import math
from contextlib import contextmanager
from tempfile import TemporaryDirectory
from tqdm import tqdm


log = logging.getLogger(__name__)


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

        r = requests.get(url, stream=True)
        total_size = int(r.headers.get("content-length", 0))
        block_size = 1024
        num_iterables = math.ceil(total_size // block_size)

        with open(filepath, "wb") as file:
            for data in tqdm(
                r.iter_content(block_size),
                total=num_iterables,
                unit="KB",
                unit_scale=True,
            ):
                file.write(data)
    else:
        log.debug("File {} already downloaded".format(filepath))
    if expected_bytes is not None:
        statinfo = os.stat(filepath)
        if statinfo.st_size != expected_bytes:
            os.remove(filepath)
            raise IOError("Failed to verify {}".format(filepath))

    return filepath


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
