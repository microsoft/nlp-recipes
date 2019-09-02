# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
    Utility functions for downloading, extracting, and reading the
    Cross-Lingual NLI Corpus (XNLI).
    https://www.nyu.edu/projects/bowman/xnli/
"""


import os
import pandas as pd

from utils_nlp.dataset.url_utils import extract_zip, maybe_download
from utils_nlp.dataset.preprocess import convert_to_unicode

URL_XNLI = "https://www.nyu.edu/projects/bowman/xnli/XNLI-1.0.zip"
URL_XNLI_MT = "https://www.nyu.edu/projects/bowman/xnli/XNLI-MT-1.0.zip"


def load_pandas_df(local_cache_path=".", file_split="dev", language="zh"):
    """Downloads and extracts the dataset files.

    Utilities information can be found `on this link <https://www.nyu.edu/projects/bowman/xnli/>`_.

    Args:
        local_cache_path (str, optional): Path to store the data.
            Defaults to "./".
        file_split (str, optional): The subset to load.
            One of: {"train", "dev", "test"}
            Defaults to "dev".
        language (str, optional): language subset to read.
            One of: {"en", "fr", "es", "de", "el", "bg", "ru",
            "tr", "ar", "vi", "th", "zh", "hi", "sw", "ur"}
            Defaults to "zh" (Chinese).
    Returns:
        pd.DataFrame: pandas DataFrame containing the specified
            XNLI subset.
    """

    if file_split in ("dev", "test"):
        url = URL_XNLI
        sentence_1_index = 6
        sentence_2_index = 7
        label_index = 1

        zip_file_name = url.split("/")[-1]
        folder_name = ".".join(zip_file_name.split(".")[:-1])
        file_name = folder_name + "/" + ".".join(["xnli", file_split, "tsv"])
    elif file_split == "train":
        url = URL_XNLI_MT
        sentence_1_index = 0
        sentence_2_index = 1
        label_index = 2

        zip_file_name = url.split("/")[-1]
        folder_name = ".".join(zip_file_name.split(".")[:-1])
        file_name = folder_name + "/multinli/" + ".".join(["multinli", file_split, language, "tsv"])

    maybe_download(url, zip_file_name, local_cache_path)

    if not os.path.exists(os.path.join(local_cache_path, folder_name)):
        extract_zip(os.path.join(local_cache_path, zip_file_name), local_cache_path)

    with open(os.path.join(local_cache_path, file_name), "r", encoding="utf-8") as f:
        lines = f.read().splitlines()

    line_list = [line.split("\t") for line in lines]

    # Remove the column name row
    line_list.pop(0)
    if file_split != "train":
        line_list = [line for line in line_list if line[0] == language]

    valid_lines = [
        True if line[sentence_1_index] and line[sentence_2_index] else False for line in line_list
    ]
    total_line_count = len(line_list)
    line_list = [line for line, valid in zip(line_list, valid_lines) if valid]
    valid_line_count = len(line_list)

    if valid_line_count != total_line_count:
        print("{} invalid lines removed.".format(total_line_count - valid_line_count))

    label_list = [convert_to_unicode(line[label_index]) for line in line_list]
    old_contradict_label = convert_to_unicode("contradictory")
    new_contradict_label = convert_to_unicode("contradiction")
    label_list = [
        new_contradict_label if label == old_contradict_label else label for label in label_list
    ]
    text_list = [
        (convert_to_unicode(line[sentence_1_index]), convert_to_unicode(line[sentence_2_index]))
        for line in line_list
    ]

    df = pd.DataFrame({"text": text_list, "label": label_list})

    return df
