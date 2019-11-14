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
import logging
import numpy as np

from tempfile import TemporaryDirectory
from utils_nlp.dataset.url_utils import extract_zip, maybe_download
from utils_nlp.models.transformers.common import MAX_SEQ_LEN
from utils_nlp.models.transformers.sequence_classification import Processor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


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


def load_tc_dataset(
    local_path=TemporaryDirectory().name,
    test_fraction=0.25,
    random_seed=None,
    train_sample_ratio=1.0,
    test_sample_ratio=1.0,
    model_name="bert-base-uncased",
    to_lower=True,
    cache_dir=TemporaryDirectory().name,
    max_len=MAX_SEQ_LEN,
    batch_size=32,
    num_gpus=None
):
    """
    Load the multinli dataset and split into training and testing datasets.
    The datasets are preprocessed and can be used to train a NER model or evaluate
    on the testing dataset.

    Args:
        local_path (str, optional): The local file path to save the raw wikigold file.
            Defautls to TemporaryDirectory().name.
        test_fraction (float, optional): The fraction of testing dataset when splitting.
            Defaults to 0.25.
        random_seed (float, optional): Random seed used to shuffle the data.
            Defaults to None.
        train_sample_ratio (float, optional): The ratio that used to sub-sampling for training.
            Defaults to 1.0.
        test_sample_ratio (float, optional): The ratio that used to sub-sampling for testing.
            Defaults to 1.0.
        model_name (str, optional): The pretained model name.
            Defaults to "bert-base-uncased".
        to_lower (bool, optional): Lower case text input.
            Defaults to True.
        cache_dir (str, optional): The default folder for saving cache files.
            Defaults to TemporaryDirectory().name.
        max_len (int, optional): Maximum length of the list of tokens. Lists longer
            than this are truncated and shorter ones are padded with "O"s. 
            Default value is BERT_MAX_LEN=512.
        batch_size (int, optional): The batch size for training and testing.
            Defaults to 32.
        num_gpus (int, optional): The number of GPUs.
            Defaults to None.

    Returns:
        tuple. The tuple contains four elements:
        train_dataload (DataLoader): a PyTorch DataLoader instance for training.

        test_dataload (DataLoader): a PyTorch DataLoader instance for testing.
        
        label_encoder (LabelEncoder): a sklearn LabelEncoder instance. The label values
            can be retrieved by calling the `inverse_transform` function.
        
        test_labels (Series): a Pandas Series of testing label (in label ID format). If
            the labels are in raw label values format, we will need to transform it to 
            label IDs by using the label_encoder.transform function.
    """

     # download and load the original dataset
    all_df = load_pandas_df(
        local_cache_path=local_path,
        num_rows=None
    )

    # set the text and label columns
    text_col = all_df.columns[0]
    label_col = all_df.columns[1]

    label_encoder = LabelEncoder()
    label_encoder.fit(["culture", "diverse", "economy", "politics", "sports"])

    # remove empty documents
    all_df = all_df[all_df[text_col].isna() == False]

    if test_fraction < 0 or test_fraction >= 1.0:
        logging.warning("Invalid test fraction value: {}, changed to 0.25".format(test_fraction))
        test_fraction = 0.25
    
    train_df, test_df = train_test_split(
        all_df,
        train_size=(1.0 - test_fraction),
        random_state=random_seed
    )

    if train_sample_ratio > 1.0:
        train_sample_ratio = 1.0
        logging.warning("Setting the training sample ratio to 1.0")
    elif train_sample_ratio < 0:
        logging.error("Invalid training sample ration: {}".format(train_sample_ratio))
        raise ValueError("Invalid training sample ration: {}".format(train_sample_ratio))
    
    if test_sample_ratio > 1.0:
        test_sample_ratio = 1.0
        logging.warning("Setting the testing sample ratio to 1.0")
    elif test_sample_ratio < 0:
        logging.error("Invalid testing sample ration: {}".format(test_sample_ratio))
        raise ValueError("Invalid testing sample ration: {}".format(test_sample_ratio))

    if train_sample_ratio < 1.0:
        train_df = train_df.sample(frac=train_sample_ratio).reset_index(drop=True)
    if test_sample_ratio < 1.0:
        test_df = test_df.sample(frac=test_sample_ratio).reset_index(drop=True)

    processor = Processor(
        model_name=model_name,
        to_lower=to_lower,
        cache_dir=cache_dir
    )

    train_dataloader = processor.create_dataloader_from_df(
        df=train_df,
        text_col=text_col,
        label_col=label_col,
        max_len=max_len,
        text2_col=None,
        batch_size=batch_size,
        num_gpus=num_gpus,
        shuffle=True,
        distributed=False
    )

    test_dataloader = processor.create_dataloader_from_df(
        df=test_df,
        text_col=text_col,
        label_col=label_col,
        max_len=max_len,
        text2_col=None,
        batch_size=batch_size,
        num_gpus=num_gpus,
        shuffle=False,
        distributed=False
    )

    # the DAC dataset already converted the labels to label ID format
    test_labels = test_df[label_col]
    return (train_dataloader, test_dataloader, label_encoder, test_labels)


def get_label_values(label_encoder, label_ids):
    """
    Get the label values from label IDs. 

    Args:
        label_encoder (LabelEncoder): a fitted sklearn LabelEncoder instance
        label_ids (Numpy array): a Numpy array of label IDs.

    Returns:
        Numpy array. A Numpy array of label values.
    """

    return label_encoder.inverse_transform(label_ids)
