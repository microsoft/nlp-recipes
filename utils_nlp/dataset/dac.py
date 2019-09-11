# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Dataset for Arabic Classification utils
https://data.mendeley.com/datasets/v524p5dhpj/2
Mohamed, BINIZ (2018), “DataSet for Arabic Classification”, Mendeley Data, v2
paper link:  ("https://www.mendeley.com/catalogue/
        arabic-text-classification-using-deep-learning-technics/")
"""

import os
import pandas as pd
from utils_nlp.dataset.url_utils import extract_zip, maybe_download

URL = (
    "https://data.mendeley.com/datasets/v524p5dhpj/2"
    "/files/91cb8398-9451-43af-88fc-041a0956ae2d/"
    "arabic_dataset_classifiction.csv.zip"
)


def load_pandas_df(local_cache_path=None, num_rows=None):
    """Downloads and extracts the dataset files
    Args:
        local_cache_path ([type], optional): [description]. Defaults to None.
        num_rows (int): Number of rows to load. If None, all data is loaded.
    Returns:
        pd.DataFrame: pandas DataFrame containing the loaded dataset.
    """
    zip_file = URL.split("/")[-1]    
    maybe_download(URL, zip_file, local_cache_path)

    zip_file_path = os.path.join(local_cache_path, zip_file)
    csv_file_path = os.path.join(local_cache_path, zip_file.replace(".zip", ""))

    if not os.path.exists(csv_file_path):
        extract_zip(file_path=zip_file_path, dest_path=local_cache_path)
    return pd.read_csv(csv_file_path, nrows=num_rows)
