# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
    Utility functions for downloading, extracting, and reading the Microsoft
    Research Paraphrase Corpus (MSRPC) dataset.
    https://www.microsoft.com/en-us/download/details.aspx?id=52398
"""

import os
import pathlib

import pandas as pd

from utils_nlp.dataset.url_utils import maybe_download, download_path

DATASET_DICT = {
    "train": "msr_paraphrase_train.txt",
    "test": "msr_paraphrase_test.txt",
    "all": "msr_paraphrase_data.txt",
}


def download_msrpc(download_dir):
    """Downloads Windows Installer for Microsoft Paraphrase Corpus.
    
    Args:
        download_dir (str): File path for the downloaded file

    Returns:
        str: file_path to the downloaded dataset.
    """

    url = (
        "https://download.microsoft.com/download/D/4/6/D46FF87A-F6B9-4252-AA8B"
        "-3604ED519838/MSRParaphraseCorpus.msi"
    )
    return maybe_download(url, work_directory=download_dir)


def load_pandas_df(local_cache_path=None, dataset_type="train"):
    """Load pandas dataframe and clean the data from the downloaded dataset

    Args:
        the dataset is already downloaded.
        dataset_type (str): Key to the DATASET_DICT item. Loads the dataset specified.
        Could be train or test.
        local_cache_path (str): Path to download the dataset installer.

    Returns:
        pd.DataFrame: A pandas dataframe with 3 columns, Sentence 1, Sentence 2 and
        score.

    """

    if dataset_type not in DATASET_DICT.keys():
        raise Exception("Dataset type not found!")

    with download_path(local_cache_path) as path:
        path = pathlib.Path(path)
        installer_datapath = download_msrpc(path)

        print(
            "The Windows Installer for Mircosoft Paraphrase Corpus has been " "downloaded at ",
            installer_datapath,
            "\n",
        )
        data_directory = input("Please install and provide the installed directory. Thanks! \n")

        data_directory = pathlib.Path(data_directory)
        assert os.path.exists(data_directory)

        fields = ["Quality", "#1 String", "#2 String"]
        file_path = os.path.join(data_directory, DATASET_DICT[dataset_type])
        df = (
            pd.read_csv(file_path, delimiter="\t", error_bad_lines=False, usecols=fields)
            .dropna()
            .rename(
                index=str,
                columns={"Quality": "score", "#1 String": "sentence1", "#2 String": "sentence2"},
            )
        )
        return df
