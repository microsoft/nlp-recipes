# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import shutil
import azureml.dataprep as dprep
import pandas as pd
from zipfile import ZipFile
from utils_nlp.dataset.url_utils import maybe_download, download_path


# Constants
SNLI_URL = "https://nlp.stanford.edu/projects/snli/snli_1.0.zip"
FILE_SPLITS = ("dev", "train", "test")
FILE_TYPES = ("jsonl", "txt")
DIR_NAMES = ("raw", "clean")
DEFAULT_FILE_TYPE = "txt"
DEFAULT_FILE_SPLIT = "train"
SNLI_FILE_PREFIX = "snli_1.0_"
SNLI_PATH = "snli_1.0"


def load_pandas_df(
    local_cache_path=None,
    file_split=DEFAULT_FILE_SPLIT,
    file_type=DEFAULT_FILE_TYPE,
):
    """
    Loads the SNLI dataset as pd.DataFrame.
    Download the dataset from "https://nlp.stanford.edu/projects/snli/snli_1.0.zip", unzip, and load.

    Args:
        local_cache_path(str): Path (directory or a zip file) to cache the downloaded zip file.
                               If None, all the intermediate files will be stored in a temporary directory and removed
                               after use.
        file_split(str): File split to load. One of (dev, test, train)
        file_type(str): File type to load. One of (txt, jsonl)

    Returns:
        pd.DataFrame: SNLI dataset.
    """
    with download_path(local_cache_path) as path:
        filepath = os.path.join(path, "snli_1.0.zip")
        snlipath = _maybe_download_and_extract(filepath, file_split, file_type)

        if file_type == FILE_TYPES[0]:
            snli_df = pd.read_json(snlipath, lines=True)
        else:
            snli_df = pd.read_csv(snlipath, sep="\t")

    return snli_df


def _maybe_download_and_extract(zip_path, file_split, file_type):
    """
    Downloads SNLI dataset zip and extract provided datafile split if they don’t already exist
    Args:
        zip_path(str): Path (directory or a zip file) to cache the downloaded zip file.
        file_split(str): File split to load. One of (dev, test, train)
        file_type: File type to load. One of (txt, jsonl)

    Returns:
         file_path: File path where data file is extracted
    """
    dirs, _ = os.path.split(zip_path)
    if not os.path.exists(dirs):
        os.makedirs(dirs)

    # store raw data here
    dir_path = os.path.join(dirs, DIR_NAMES[0], "snli_1.0")

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # csv file
    file_name = SNLI_FILE_PREFIX + file_split + "." + file_type
    file_path = os.path.join(dir_path, file_name)
    source_path = SNLI_PATH + "/" + file_name

    if not os.path.exists(file_path):
        download_snli(zip_path)
        extract_snli(zip_path, source_path, dest_path=file_path)
    return file_path


def download_snli(dest_path):
    """
        Download the SNLI dataset
    Args:
        dest_path: file path where SNLI dataset should be downloaded

    Returns:
        file_path: file path where SNLI dataset is downloaded

    """
    dirs, file = os.path.split(dest_path)
    maybe_download(SNLI_URL, file, work_directory=dirs)


def extract_snli(zip_path, source_file_name, dest_path):
    """
    Extract SNLI datafile from the SNLI raw zip file.
    Args:
        zip_path: zip file location
        source_file_name: datafile location.
        dest_path: file path where SNLI should be extracted.

    """
    with ZipFile(zip_path, "r") as z:
        with z.open(source_file_name) as zf, open(dest_path, "wb") as f:
            shutil.copyfileobj(zf, f)


def clean_snli(source_file_path):
    """
        Remove the extra columns from the input dataframe
    Args:
        file_path: remove columns from the given file

    Returns:
        pd.DataFrame : pandas Dataframe
    """
    source_filename, source_file_extension = os.path.splitext(source_file_path)

    if source_file_extension == FILE_TYPES[0]:
        snli_df = pd.read_json(source_file_path, lines=True)
    else:
        snli_df = pd.read_csv(source_file_path, sep="\t")

    snli_df = snli_df.drop(
        [
            "sentence1_binary_parse",
            "sentence2_binary_parse",
            "sentence1_parse",
            "sentence2_parse",
            "captionID",
            "pairID",
            "label1",
            "label2",
            "label3",
            "label4",
            "label5",
        ],
        axis=1,
    )

    snli_df = snli_df.rename(index=str, columns={"gold_label": "score"})

    return snli_df


def load_azureml_df(
    local_cache_path=None,
    file_split=DEFAULT_FILE_SPLIT,
    file_type=DEFAULT_FILE_TYPE,
):
    """
    Loads the SNLI dataset as AzureML dataflow object.
    Download the dataset from "https://nlp.stanford.edu/projects/snli/snli_1.0.zip", unzip, and load.

    Args:
        local_cache_path(str): Path (directory or a zip file) to cache the downloaded zip file.
                               If None, all the intermediate files will be stored in a temporary directory and removed
                               after use.
        file_split(str): File split to load. One of (dev, test, train)
        file_type(str): File type to load. One of (txt, jsonl)

    Returns:
        AzureML dataflow object: SNLI dataset.

    """
    with download_path(local_cache_path) as path:
        filepath = os.path.join(path, "snli_1.0.zip")
        snlipath = _maybe_download_and_extract(filepath, file_split, file_type)

        # this does not correctly convert the .jsonl file.
        df = dprep.auto_read_file(snlipath)

    return df
