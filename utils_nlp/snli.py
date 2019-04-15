# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import shutil
import azureml.dataprep as dprep
import pandas as pd
from zipfile import ZipFile
from utils_nlp.url_utils import maybe_download, download_path
from utils_nlp.snli_constants import (
    DEFAULT_GOLD_COL,
    DEFAULT_SENTENCE_ONE_BINARY_PARSE_COL,
    DEFAULT_SENTENCE_TWO_BINARY_PARSE_COL,
    DEFAULT_SENTENCE_ONE_PARSE_COL,
    DEFAULT_SENTENCE_TWO_PARSE_COL,
    DEFAULT_SENTENCE_ONE_COL,
    DEFAULT_SENTENCE_TWO_COL,
    DEFAULT_CAPTION_ID_COL,
    DEFAULT_PAIR_ID_COL,
    DEFAULT_LABEL_ONE_COL,
    DEFAULT_LABEL_TWO_COL,
    DEFAULT_LABEL_THREE_COL,
    DEFAULT_LABEL_FOUR_COL,
    DEFAULT_LABEL_FIVE_COL
)

DEFAULT_HEADER = (
    DEFAULT_GOLD_COL,
    DEFAULT_SENTENCE_ONE_BINARY_PARSE_COL,
    DEFAULT_SENTENCE_TWO_BINARY_PARSE_COL,
    DEFAULT_SENTENCE_ONE_PARSE_COL,
    DEFAULT_SENTENCE_TWO_PARSE_COL,
    DEFAULT_SENTENCE_ONE_COL,
    DEFAULT_SENTENCE_TWO_COL,
    DEFAULT_CAPTION_ID_COL,
    DEFAULT_PAIR_ID_COL,
    DEFAULT_LABEL_ONE_COL,
    DEFAULT_LABEL_TWO_COL,
    DEFAULT_LABEL_THREE_COL,
    DEFAULT_LABEL_FOUR_COL,
    DEFAULT_LABEL_FIVE_COL
)

SNLI_JSONL_FILE_NAMES = (
    "snli_1.0_dev.jsonl",
    "snli_1.0_train.jsonl",
    "snli_1.0_test.jsonl"
)

SNLI_TEXT_FILE_NAMES = (
    "snli_1.0_dev.txt",
    "snli_1.0_train.txt",
    "snli_1.0_test.txt",
)


def load_azureml_df(
 #   header=DEFAULT_HEADER,
    local_cache_path=None
):
    with download_path(local_cache_path) as path:
        filepath = os.path.join(path, "snli_1.0.zip")
        snlipath = _maybe_download_and_extract(filepath)

        df = dprep.auto_read_file(snlipath)

        df.head(5)
    return df


def load_pandas_df(
 #   header=DEFAULT_HEADER,
    local_cache_path=None
):
    with download_path(local_cache_path) as path:
        filepath = os.path.join(path, "snli_1.0.zip")
        snlipath = _maybe_download_and_extract(filepath)

        #df = pd.read_csv(snlipath, sep='\t')
        df = pd.read_json(snlipath, lines=True)

        df.head(5)
    return df


def _maybe_download_and_extract(zip_path):
    """Downloads and extracts snli txt and jsonl datafiles if they don’t already exist"""
    dirs, _ = os.path.split(zip_path)
    if not os.path.exists(dirs):
        os.makedirs(dirs)

    dir_path = os.path.dirname(zip_path)

    # csv file
    #file_path = os.path.join(dir_path,  SNLI_TEXT_FILE_NAMES[0])#TO-DO

    # jsonl file
    file_path = os.path.join(dir_path,  SNLI_JSONL_FILE_NAMES[0])#TO-DO

    if not os.path.exists(file_path):
        download_snli(zip_path)
        extract_snli(zip_path, dest_path=dir_path)
    return file_path


def download_snli(dest_path):
    url = "https://nlp.stanford.edu/projects/snli/snli_1.0.zip"
    dirs, file = os.path.split(dest_path)
    maybe_download(url, file, work_directory=dirs)


def extract_snli(zip_path, dest_path):
    #read_txt_file(zip_path, dest_path)
    read_jsonl_file(zip_path, dest_path)

def read_txt_file(zip_path, dest_path):
    with ZipFile(zip_path, "r") as z:
        with z.open("snli_1.0/snli_1.0_dev.txt") as zf, open("C:/NLP/data/snli_1.0_dev.txt", "wb") as f:
            shutil.copyfileobj(zf, f)


def read_jsonl_file(zip_path, dest_path):
    with ZipFile(zip_path, "r") as z:
        with z.open("snli_1.0/snli_1.0_dev.jsonl") as zf, open("C:/NLP/data/snli_1.0_dev.jsonl", "wb") as f:
            shutil.copyfileobj(zf, f)

