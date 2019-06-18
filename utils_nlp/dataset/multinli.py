# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""MultiNLI dataset utils
https://www.nyu.edu/projects/bowman/multinli/
"""

import os
import pandas as pd
from utils_nlp.dataset.url_utils import extract_zip, maybe_download

URL = "http://www.nyu.edu/projects/bowman/multinli/multinli_1.0.zip"
DATA_FILES = {
    "train": "multinli_1.0/multinli_1.0_train.jsonl",
    "dev_matched": "multinli_1.0/multinli_1.0_dev_matched.jsonl",
    "dev_mismatched": "multinli_1.0/multinli_1.0_dev_mismatched.jsonl",
}


def load_pandas_df(local_cache_path=None, file_split="train"):
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

    if not os.path.exists(
        os.path.join(local_cache_path, DATA_FILES[file_split])
    ):
        extract_zip(
            os.path.join(local_cache_path, file_name), local_cache_path
        )
    return pd.read_json(
        os.path.join(local_cache_path, DATA_FILES[file_split]), lines=True
    )
