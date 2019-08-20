# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
    Utility functions for downloading, extracting, and reading the
    STSbenchmark dataset.
    http://ixa2.si.ehu.es/stswiki/index.php/STSbenchmark
"""


import os
import tarfile
import pandas as pd

from utils_nlp.dataset.url_utils import maybe_download

STS_URL = "http://ixa2.si.ehu.es/stswiki/images/4/48/Stsbenchmark.tar.gz"
DEFAULT_FILE_SPLIT = "train"


def load_pandas_df(data_path, file_split=DEFAULT_FILE_SPLIT):
    """Load the STS Benchmark dataset as a pd.DataFrame

    Args:
        data_path (str): Path to data directory
        file_split (str, optional): File split to load.
        One of (train, dev, test).
        Defaults to train.

    Returns:
        pd.DataFrame: STS Benchmark dataset
    """
    file_name = "sts-{}.csv".format(file_split)
    df = _maybe_download_and_extract(file_name, data_path)
    return df


def _maybe_download_and_extract(sts_file, base_data_path):
    raw_data_path = os.path.join(base_data_path, "raw")
    if not os.path.exists(raw_data_path):
        os.makedirs(raw_data_path)
    sts_path = _download_sts(raw_data_path)
    df = _load_sts(os.path.join(sts_path, sts_file))
    return df


def _download_sts(dirpath):
    """Download and extract data from
        http://ixa2.si.ehu.es/stswiki/images/4/48/Stsbenchmark.tar.gz

    Args:
        dirpath (str): Path to data directory.

    Returns:
        str: Path to extracted STS Benchmark data.
    """
    filepath = maybe_download(STS_URL, work_directory=dirpath)
    extracted_path = _extract_sts(filepath, target_dirpath=dirpath, tmode="r:gz")
    print("Data downloaded to {}".format(extracted_path))
    return extracted_path


def _extract_sts(tarpath, target_dirpath=".", tmode="r"):
    """Extract data from the sts tar.gz archive

    Args:
        tarpath (str): Path to tarfile, to be deleted after extraction.
        target_dirpath (str, optional): Directory in which to save
            the extracted files.
        tmode (str, optional): The mode for reading,
            of the form "filemode[:compression]".
        Defaults to "r".

    Returns:
        str: Path to extracted STS Benchmark data.
    """
    with tarfile.open(tarpath, mode=tmode) as t:
        t.extractall(target_dirpath)
        extracted = t.getnames()[0]
    os.remove(tarpath)
    return os.path.join(target_dirpath, extracted)


def _load_sts(src_file_path):
    """Load datafile as dataframe

    Args:
        src_file_path (str): filepath to train/dev/test csv files.
    """
    with open(src_file_path, "r", encoding="utf-8") as f:
        sent_pairs = []
        for line in f:
            line = line.strip().split("\t")
            sent_pairs.append(
                [
                    line[0].strip(),
                    line[1].strip(),
                    line[2].strip(),
                    line[3].strip(),
                    float(line[4]),
                    line[5].strip(),
                    line[6].strip(),
                ]
            )

        sdf = pd.DataFrame(
            sent_pairs,
            columns=[
                "column_0",
                "column_1",
                "column_2",
                "column_3",
                "column_4",
                "column_5",
                "column_6",
            ],
        )
        return sdf


def clean_sts(df):
    """Drop columns containing irrelevant metadata and
    save as new csv files in the target_dir.

    Args:
        df (pandas.Dataframe): drop columns from train/test/dev files.
    """
    clean_df = df.drop(["column_0", "column_1", "column_2", "column_3"], axis=1)
    clean_df = clean_df.rename(
        index=str, columns={"column_4": "score", "column_5": "sentence1", "column_6": "sentence2"}
    )
    return clean_df
