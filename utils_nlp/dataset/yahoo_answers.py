# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Yahoo! Answers dataset utils"""

import os
import pandas as pd
from utils_nlp.dataset.url_utils import maybe_download, extract_tar


URL = "https://s3.amazonaws.com/fast-ai-nlp/yahoo_answers_csv.tgz"


def download(dir_path):
    """Downloads and extracts the dataset files"""
    file_name = URL.split("/")[-1]
    maybe_download(URL, file_name, dir_path)
    extract_tar(os.path.join(dir_path, file_name), dir_path)


def read_data(data_file, nrows=None):
    return pd.read_csv(data_file, header=None, nrows=nrows)


def get_text(df):
    df.fillna("", inplace=True)
    text = df.iloc[:, 1] + " " + df.iloc[:, 2] + " " + df.iloc[:, 3]
    text = text.str.replace(r"[^A-Za-z ]", "").str.lower()
    text = text.str.replace(r"\\s+", " ")
    text = text.astype(str)
    return text


def get_labels(df):
    return list(df[0] - 1)
