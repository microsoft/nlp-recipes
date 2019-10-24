# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
    Utility functions for downloading, extracting, and reading the
    Multi-Genre NLI (MultiNLI) Corpus.
    https://www.nyu.edu/projects/bowman/multinli/
"""

import os

import pandas as pd

from utils_nlp.dataset.data_loaders import DaskJSONLoader
from utils_nlp.dataset.url_utils import extract_zip, maybe_download

URL = "http://www.nyu.edu/projects/bowman/multinli/multinli_1.0.zip"
DATA_FILES = {
    "train": "multinli_1.0/multinli_1.0_train.jsonl",
    "dev_matched": "multinli_1.0/multinli_1.0_dev_matched.jsonl",
    "dev_mismatched": "multinli_1.0/multinli_1.0_dev_mismatched.jsonl",
}
SAMPLE_URL = "https://nlpbp.blob.core.windows.net/data/mnli_sample.csv"


def download_file_and_extract(local_cache_path: str = ".") -> None:
    file_name = SAMPLE_URL.split("/")[-1]
    maybe_download(SAMPLE_URL, file_name, local_cache_path)
    return file_name


def load_pandas_df(local_cache_path="."):
    try:
        file_name = download_file_and_extract(local_cache_path)        
    except Exception as e:
        raise e
    return pd.read_csv(os.path.join(local_cache_path, file_name))


def get_generator(
    local_cache_path=".", file_split="train", block_size=10e6, batch_size=10e6, num_batches=None
):
    """ Returns an extracted dataset as a random batch generator that
    yields pandas dataframes.
    Args:
        local_cache_path ([type], optional): [description]. Defaults to None.
        file_split (str, optional): The subset to load.
            One of: {"train", "dev_matched", "dev_mismatched"}
            Defaults to "train".
        block_size (int, optional): Size of partition in bytes.
        num_batches (int): Number of batches to generate.
        batch_size (int]): Batch size.
    Returns:
        Generator[pd.Dataframe, None, None] : Random batch generator that yields pandas dataframes.
    """

    try:
        download_file_and_extract(local_cache_path, file_split)
    except Exception as e:
        raise e

    loader = DaskJSONLoader(
        os.path.join(local_cache_path, DATA_FILES[file_split]), block_size=block_size
    )

    return loader.get_sequential_batches(batch_size=int(batch_size), num_batches=num_batches)
