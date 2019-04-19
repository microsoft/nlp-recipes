# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import shutil
import azureml.dataprep as dprep
import pandas as pd
from zipfile import ZipFile
from utils_nlp.dataset.url_utils import maybe_download, download_path

FILE_SPLITS = ('dev', 'train', 'test')
FILE_TYPES = ('jsonl', 'txt')
DIR_NAMES = ('raw', 'clean')
DEFAULT_FILE_TYPE = 'txt'
DEFAULT_FILE_SPLIT = 'train'
SNLI_FILE_PREFIX = 'snli_1.0_'
SNLI_PATH = 'snli_1.0'


def load_azureml_df(local_cache_path=None, file_split=DEFAULT_FILE_SPLIT, file_type=DEFAULT_FILE_TYPE):
    with download_path(local_cache_path) as path:
        filepath = os.path.join(path, "snli_1.0.zip")
        snlipath = _maybe_download_and_extract(filepath, file_split, file_type)

        #this does not work correctly convert the .jsonl file.
        df = dprep.auto_read_file(snlipath)

    return df.to_pandas_dataframe()


def load_pandas_df(local_cache_path=None,file_split=DEFAULT_FILE_SPLIT, file_type=DEFAULT_FILE_TYPE):
    with download_path(local_cache_path) as path:
        filepath = os.path.join(path, "snli_1.0.zip")
        snlipath = _maybe_download_and_extract(filepath, file_split, file_type)

## if invalid filetype passed then handle
        if file_type == FILE_TYPES[0]:
            snli_df = pd.read_json(snlipath, lines=True)
        else:
            snli_df = pd.read_csv(snlipath, sep='\t')

    return snli_df


def _maybe_download_and_extract(zip_path, file_split, file_type):
    """Downloads and extracts snli txt and jsonl datafiles if they don’t already exist"""
    dirs, _ = os.path.split(zip_path)
    if not os.path.exists(dirs):
        os.makedirs(dirs)

    #store raw data here
    dir_path = os.path.join(dirs, DIR_NAMES[0])

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # csv file
    file_name = SNLI_FILE_PREFIX + file_split + '.' + file_type
    file_path = os.path.join(dir_path, file_name) #TO-DO - check for all files here
    source_path = SNLI_PATH + '/' + file_name # TO-DO fix this

    if not os.path.exists(file_path):
        download_snli(zip_path)
        extract_snli(zip_path, source_path, dest_path=file_path)
    return file_path


def download_snli(dest_path):
    url = "https://nlp.stanford.edu/projects/snli/snli_1.0.zip"
    dirs, file = os.path.split(dest_path)
    maybe_download(url, file, work_directory=dirs)


def extract_snli(zip_path, source_file_name, dest_path):
    with ZipFile(zip_path, 'r') as z:
        with z.open(source_file_name) as zf, open(dest_path, "wb") as f:
            shutil.copyfileobj(zf, f)


def clean_snli(snli_df):
    snli_df = snli_df.drop(['sentence1_binary_parse', 'sentence2_binary_parse', 'sentence1_parse', 'sentence2_parse',
                              'captionID', 'pairID', 'label1', 'label2', 'label3', 'label4', 'label5'], axis=1)
    snli_df = snli_df.rename(index=str, columns={"gold_label": "score"})

    save_df_to_csv_file(snli_df)

    return snli_df


def save_df_to_csv_file(snli_df, file_name= SNLI_FILE_PREFIX):

    data_dir_path = os.path.dirname(os.path.realpath(__file__))
    clean_dir_path = os.path.join(data_dir_path, DIR_NAMES[1])

    if not os.path.exists(clean_dir_path):
        os.makedirs(clean_dir_path)

    file_path = os.path.join(clean_dir_path)
    snli_df.to_csv(os.path.join(clean_dir_path + "test.csv"), index=False)

