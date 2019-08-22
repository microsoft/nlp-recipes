# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
    Utility functions for downloading, extracting, and reading the Stanford
    Natural Language Inference (SNLI) Corpus.
    https://nlp.stanford.edu/projects/snli/
"""
import os
import shutil
import azureml.dataprep as dprep
import pandas as pd
from zipfile import ZipFile
from utils_nlp.dataset.url_utils import maybe_download, download_path
from utils_nlp.dataset import Split

# constants
SNLI_URL = "https://nlp.stanford.edu/projects/snli/snli_1.0.zip"
SNLI_DIRNAME = "snli_1.0"
SNLI_FILE_PREFIX = "snli_1.0"

# clean col names
S1_COL = "sentence1"
S2_COL = "sentence2"
LABEL_COL = "score"


def load_pandas_df(local_cache_path=None, file_split=Split.TRAIN, file_type="txt", nrows=None):
    """
    Loads the SNLI dataset as pd.DataFrame
    Download the dataset from "https://nlp.stanford.edu/projects/snli/snli_1.0.zip", unzip, and load

    Args:
        local_cache_path (str): Path (directory or a zip file) to cache the downloaded zip file.
            If None, all the intermediate files will be stored in a temporary directory and removed
            after use.
        file_split (str): File split to load, defaults to "train"
        file_type (str): File type to load, defaults to "txt"
        nrows (int): Number of rows to load, defaults to None (in which all rows will be returned)

    Returns:
        pd.DataFrame: SNLI dataset.
    """
    with download_path(local_cache_path) as path:
        filepath = os.path.join(path, "snli_1.0.zip")
        snlipath = _maybe_download_and_extract(filepath, file_split, file_type)

        if file_type == "txt":
            snli_df = pd.read_csv(snlipath, sep="\t", nrows=nrows)
        else:
            snli_df = pd.read_json(snlipath, lines=True)
            if nrows:
                snli_df = snli_df[:nrows]

    return snli_df


def _maybe_download_and_extract(zip_path, file_split, file_type):
    """
    Downloads SNLI dataset zip and extract provided datafile split if they don’t already exist
    Args:
        zip_path (str): Path (directory or a zip file) to cache the downloaded zip file
        file_split (str): File split to load
        file_type(str) : File type to load

    Returns:
         str: File path where data file is extracted
    """
    dirs, _ = os.path.split(zip_path)
    if not os.path.exists(dirs):
        os.makedirs(dirs)

    # store raw data here
    dir_path = os.path.join(dirs, "raw", SNLI_DIRNAME)

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # format csv filename
    file_name = "{0}_{1}.{2}".format(SNLI_FILE_PREFIX, file_split.value, file_type)
    extract_path = os.path.join(dir_path, file_name)

    if not os.path.exists(extract_path):
        _ = download_snli(zip_path)
        extract_snli(zip_path, source_path=SNLI_DIRNAME + "/" + file_name, dest_path=extract_path)

    return extract_path


def download_snli(dest_path):
    """
    Download the SNLI dataset
    Args:
        dest_path (str): file path where SNLI dataset should be downloaded

    Returns:
        str: file path where SNLI dataset is downloaded

    """
    dirs, file = os.path.split(dest_path)
    maybe_download(SNLI_URL, file, work_directory=dirs)


def extract_snli(zip_path, source_path, dest_path):
    """
    Extract SNLI datafile from the SNLI raw zip file.
    Args:
        zip_path (str): zip file location
        source_path (str): datafile location
        dest_path (str): file path for extracted SNLI

    """
    with ZipFile(zip_path, "r") as z:
        with z.open(source_path) as zf, open(dest_path, "wb") as f:
            shutil.copyfileobj(zf, f)


def clean_cols(df):
    """
    Drop irrelevant columns from the input dataframe
    Args:
        df(pd.DataFrame): Input dataframe

    Returns:
        pd.DataFrame
    """
    snli_df = df.drop(
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

    snli_df = snli_df.rename(
        columns={"sentence1": S1_COL, "sentence2": S2_COL, "gold_label": LABEL_COL}
    )

    return snli_df


def clean_rows(df, label_col=LABEL_COL):
    """Drop badly formatted rows from the input dataframe

    Args:
        df (pd.DataFrame): Input dataframe
        label_col (str): Name of label column.
            Defaults to the standardized column name that is set after running the clean_col method.

    Returns:
        pd.DataFrame
    """
    snli_df = df.dropna()
    snli_df = snli_df.loc[snli_df[label_col] != "-"].copy()

    return snli_df


def clean_df(df, label_col=LABEL_COL):
    df = clean_cols(df)
    df = clean_rows(df, label_col)

    return df


def load_azureml_df(local_cache_path=None, file_split=Split.TRAIN, file_type="txt"):
    """
    Loads the SNLI dataset as AzureML dataflow object
    Download the dataset from "https://nlp.stanford.edu/projects/snli/snli_1.0.zip", unzip,
    and load.

    Args:
        local_cache_path (str): Path (directory or a zip file) to cache the downloaded zip file.
            If None, all the intermediate files will be stored in a temporary directory and removed
            after use.
        file_split (str): File split to load. One of (dev, test, train)
        file_type (str): File type to load. One of (txt, jsonl)

    Returns:
        AzureML dataflow: SNLI dataset

    """
    with download_path(local_cache_path) as path:
        filepath = os.path.join(path, "snli_1.0.zip")
        snlipath = _maybe_download_and_extract(filepath, file_split, file_type)

        # NOTE: this works for the txt format but not the jsonl format
        df = dprep.auto_read_file(snlipath)

    return df
