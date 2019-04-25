# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import tarfile
import pandas as pd
import azureml.dataprep as dp

from utils_nlp.dataset.url_utils import maybe_download


def download_sts(dirpath):
    """Download and extract data from http://ixa2.si.ehu.es/stswiki/images/4/48/Stsbenchmark.tar.gz 

    Args:
        dirpath (str): Path to data directory.

    Returns:
        str: Path to extracted STS Benchmark data.
    """
    sts_url = "http://ixa2.si.ehu.es/stswiki/images/4/48/Stsbenchmark.tar.gz"
    filepath = maybe_download(sts_url, work_directory=dirpath)
    extracted_path = extract_sts(
        filepath, target_dirpath=dirpath, tmode="r:gz"
    )
    print("Data downloaded to {}".format(extracted_path))
    return extracted_path


def extract_sts(tarpath, target_dirpath=".", tmode="r"):
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


def clean_sts(filenames, src_dir, target_dir):
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


class STSBenchmark:
    def __init__(self, which_split, base_data_path="./data"):
        """Download and extract the data if it does not already exist in the base data directory

        Args:
            which_split (str): Either "train", "test", or "dev". 
            base_data_path (str, optional): Base data directory.
        """
        assert which_split in set(["train", "test", "dev"])
        self.base_data_path = base_data_path
        self.filepath = os.path.join(
            self.base_data_path,
            "clean",
            "stsbenchmark",
            "sts-{}.csv".format(which_split),
        )
        self._maybe_download_and_extract()

    def _maybe_download_and_extract(self):
        """ Check if a clean dataframe for the specified split exists. If not, download the entire dataset and clean """
        if not os.path.exists(self.filepath):
            raw_path = os.path.join(self.base_data_path, "raw")
            if not os.path.exists(raw_path):
                os.makedirs(raw_path)
            sts_path = download_sts(raw_path)
            sts_files = [f for f in os.listdir(sts_path) if f.endswith(".csv")]
            clean_sts(
                sts_files,
                sts_path,
                os.path.join(self.base_data_path, "clean", "stsbenchmark"),
            )

    def as_dataframe(self):
        """Return the clean data as a pandas dataframe 

        Returns:
            pandas dataframe: Clean STS Benchmark data for the desired split.
        """
        return (
            dp.auto_read_file(self.filepath)
            .drop_columns("Column1")
            .to_pandas_dataframe()
        )
