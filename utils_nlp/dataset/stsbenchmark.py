# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import tarfile
import pandas as pd
import azureml.dataprep as dp

from utils_nlp.dataset.url_utils import maybe_download

STS_URL = "http://ixa2.si.ehu.es/stswiki/images/4/48/Stsbenchmark.tar.gz"
DEFAULT_FILE_SPLIT = "train"


def load_pandas_df(data_path, file_split=DEFAULT_FILE_SPLIT):
    """Load the STS Benchmark dataset as a pandas dataframe
    
    Args:
        data_path (str): Path to data directory
        file_split (str, optional): File split to load. One of (train, dev, test). Defaults to train.
    
    Returns:
        pd.DataFrame: STS Benchmark dataset
    """
    clean_file_path = os.path.join(
        data_path, "clean/stsbenchmark", "sts-{}.csv".format(file_split)
    )
    dflow = _maybe_download_and_extract(data_path, clean_file_path)
    return dflow.to_pandas_dataframe()


def _maybe_download_and_extract(base_data_path, clean_file_path):
    if not os.path.exists(clean_file_path):
        raw_data_path = os.path.join(base_data_path, "raw")
        if not os.path.exists(raw_data_path):
            os.makedirs(raw_data_path)
        sts_path = _download_sts(raw_data_path)
        sts_files = [f for f in os.listdir(sts_path) if f.endswith(".csv")]
        _clean_sts(
            sts_files,
            sts_path,
            os.path.join(base_data_path, "clean", "stsbenchmark"),
        )
    return dp.auto_read_file(clean_file_path).drop_columns("Column1")


def _download_sts(dirpath):
    """Download and extract data from http://ixa2.si.ehu.es/stswiki/images/4/48/Stsbenchmark.tar.gz 

    Args:
        dirpath (str): Path to data directory.

    Returns:
        str: Path to extracted STS Benchmark data.
    """
    filepath = maybe_download(STS_URL, work_directory=dirpath)
    extracted_path = _extract_sts(
        filepath, target_dirpath=dirpath, tmode="r:gz"
    )
    print("Data downloaded to {}".format(extracted_path))
    return extracted_path


def _extract_sts(tarpath, target_dirpath=".", tmode="r"):
    """Extract data from the sts tar.gz archive

    Args:
        tarpath (str): Path to tarfile, to be deleted after extraction.
        target_dirpath (str, optional): Directory in which to save the extracted files. 
        tmode (str, optional): The mode for reading, of the form "filemode[:compression]". Defaults to "r".

    Returns:
        str: Path to extracted STS Benchmark data.
    """
    with tarfile.open(tarpath, mode=tmode) as t:
        t.extractall(target_dirpath)
        extracted = t.getnames()[0]
    os.remove(tarpath)
    return os.path.join(target_dirpath, extracted)


def _clean_sts(filenames, src_dir, target_dir):
    """Drop columns containing irrelevant metadata and save as new csv files in the target_dir

    Args:
        filenames (list of str): List of filenames for the train/dev/test csv files.
        src_dir (str): Directory for the raw csv files.
        target_dir (str): Directory for the clean csv files to be written to.
    """
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    filepaths = [os.path.join(src_dir, f) for f in filenames]
    for i, fp in enumerate(filepaths):
        dat = dp.auto_read_file(path=fp)
        s = dat.keep_columns(["Column5", "Column6", "Column7"]).rename_columns(
            {
                "Column5": "score",
                "Column6": "sentence1",
                "Column7": "sentence2",
            }
        )
        print(
            "Writing clean dataframe to {}".format(
                os.path.join(target_dir, filenames[i])
            )
        )
        sdf = s.to_pandas_dataframe().to_csv(
            os.path.join(target_dir, filenames[i]), sep="\t"
        )
