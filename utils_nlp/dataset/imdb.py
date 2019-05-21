# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""IMDB dataset utils"""

import os
import pandas as pd
from utils_nlp.dataset.url_utils import maybe_download, extract_tar


URL = "https://s3.amazonaws.com/fast-ai-nlp/imdb.tgz"


def download(dir_path):
    """Downloads and extracts the dataset files"""
    file_name = URL.split("/")[-1]
    maybe_download(URL, file_name, dir_path)
    extract_tar(os.path.join(dir_path, file_name))


def get_df(dir_path, label):
    """Returns a pandas dataframe given a path,
       and appends the provided label"""
    text = []
    for doc_file in os.listdir(dir_path):
        with open(os.path.join(dir_path, doc_file)) as f:
            text.append(f.read())
    labels = [label] * len(text)
    return pd.DataFrame({"text": text, "label": labels})
