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


def load_pandas_df(local_cache_path=".", file_split="train"):
    """Downloads and extracts the dataset files
    Args:
        local_cache_path ([type], optional): [description]. Defaults to None.
        file_split (str, optional): The subset to load.
            One of: {"train", "dev_matched", "dev_mismatched"}
            Defaults to "train".
    Returns:
        pd.DataFrame: pandas DataFrame containing the specified
            MultiNLI subset.
    """

    file_name = URL.split("/")[-1]
    maybe_download(URL, file_name, local_cache_path)

    if not os.path.exists(os.path.join(local_cache_path, DATA_FILES[file_split])):
        extract_zip(os.path.join(local_cache_path, file_name), local_cache_path)
    return pd.read_json(os.path.join(local_cache_path, DATA_FILES[file_split]), lines=True)


def get_generator(
    local_cache_path=".", file_split="train", block_size=10e6, batch_size=10e6, num_batches=None
):
    """ Downloads and extracts the dataset files and then returns a random batch generator that
    yields pandas dataframes.
    Args:
        local_cache_path ([type], optional): [description]. Defaults to None.
        file_split (str, optional): The subset to load.
            One of: {"train", "dev_matched", "dev_mismatched"}
            Defaults to "train".
        block_size (int, optional): Size of partition in bytes.
        random_seed (int, optional): Random seed. See random.seed().Defaults to None.
        num_batches (int): Number of batches to generate.
        batch_size (int]): Batch size.
    Returns:
        Generator[pd.Dataframe, None, None] : Random batch generator that yields pandas dataframes.
    """

    file_name = URL.split("/")[-1]
    maybe_download(URL, file_name, local_cache_path)

    if not os.path.exists(os.path.join(local_cache_path, DATA_FILES[file_split])):
        extract_zip(os.path.join(local_cache_path, file_name), local_cache_path)

    loader = DaskJSONLoader(
        os.path.join(local_cache_path, DATA_FILES[file_split]), block_size=block_size
    )

    return loader.get_sequential_batches(batch_size=int(batch_size), num_batches=num_batches)
