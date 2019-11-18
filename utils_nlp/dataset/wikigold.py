# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
    Utility functions for downloading and reading the wikigold dataset for
    Named Entity Recognition (NER).
    https://github.com/juand-r/entity-recognition-datasets/tree/master/data/wikigold/CONLL-format/data
"""

import random
import os
import pandas as pd
import logging

from tempfile import TemporaryDirectory
from utils_nlp.dataset.url_utils import maybe_download
from utils_nlp.dataset.ner_utils import preprocess_conll
from utils_nlp.models.transformers.common import MAX_SEQ_LEN
from utils_nlp.models.transformers.named_entity_recognition import TokenClassificationProcessor


URL = (
    "https://raw.githubusercontent.com/juand-r/entity-recognition-datasets"
    "/master/data/wikigold/CONLL-format/data/wikigold.conll.txt"
)


def load_train_test_dfs(local_cache_path="./", test_fraction=0.5, random_seed=None):
    """
    Get the training and testing data frames based on test_fraction.

    Args:
        local_cache_path (str): Path to store the data. If the data file
            doesn't exist in this path, it's downloaded.
        test_fraction (float, optional): Fraction of data ot use for
            testing. Since this is a small dataset, the default testing
            fraction is set to 0.5
        random_seed (float, optional): Random seed used to shuffle the data.

    Returns:
        tuple: (train_pandas_df, test_pandas_df), each data frame contains
            two columns
            "sentence": sentences in strings.
            "labels": list of entity labels of the words in the sentence.

    """
    file_name = URL.split("/")[-1]
    maybe_download(URL, file_name, local_cache_path)

    data_file = os.path.join(local_cache_path, file_name)

    with open(data_file, "r", encoding="utf8") as file:
        text = file.read()

    sentence_list, labels_list = preprocess_conll(text)

    if random_seed:
        random.seed(random_seed)
    sentence_and_labels = list(zip(sentence_list, labels_list))
    random.shuffle(sentence_and_labels)
    sentence_list[:], labels_list[:] = zip(*sentence_and_labels)

    sentence_count = len(sentence_list)
    test_sentence_count = round(sentence_count * test_fraction)
    test_sentence_list = sentence_list[:test_sentence_count]
    test_labels_list = labels_list[:test_sentence_count]
    train_sentence_list = sentence_list[test_sentence_count:]
    train_labels_list = labels_list[test_sentence_count:]

    train_df = pd.DataFrame({"sentence": train_sentence_list, "labels": train_labels_list})

    test_df = pd.DataFrame({"sentence": test_sentence_list, "labels": test_labels_list})

    return (train_df, test_df)


def get_unique_labels():
    """Get the unique labels in the wikigold dataset."""
    return ["O", "I-LOC", "I-MISC", "I-PER", "I-ORG"]


def load_dataset(
    local_path=TemporaryDirectory().name,
    test_fraction=0.3,
    random_seed=None,
    train_sample_ratio=1.0,
    test_sample_ratio=1.0,
    model_name="bert-base-uncased",
    to_lower=True,
    cache_dir=TemporaryDirectory().name,
    max_len=MAX_SEQ_LEN,
    trailing_piece_tag="X",
    batch_size=32,
    num_gpus=None
):
    """
    Load the wikigold dataset and split into training and testing datasets.
    The datasets are preprocessed and can be used to train a NER model or evaluate
    on the testing dataset.

    Args:
        local_path (str, optional): The local file path to save the raw wikigold file.
            Defautls to "~/.nlp_utils/datasets/".
        test_fraction (float, optional): The fraction of testing dataset when splitting.
            Defaults to 0.3.
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
            Defaults to './temp'.
        max_len (int, optional): Maximum length of the list of tokens. Lists longer
            than this are truncated and shorter ones are padded with "O"s. 
            Default value is BERT_MAX_LEN=512.
        trailing_piece_tag (str, optional): Tag used to label trailing word pieces.
            For example, "criticize" is broken into "critic" and "##ize", "critic"
            preserves its original label and "##ize" is labeled as trailing_piece_tag.
            Default value is "X".
        batch_size (int, optional): The batch size for training and testing.
            Defaults to 32.
        num_gpus (int, optional): The number of GPUs.
            Defaults to None.

    Returns:
        tuple. The tuple contains four elements.
        train_dataload (DataLoader): a PyTorch DataLoader instance for training.

        test_dataload (DataLoader): a PyTorch DataLoader instance for testing.
        
        label_map (dict): A dictionary object to map a label (str) to an ID (int). 

        test_dataset (TensorDataset): A TensorDataset containing the following four tensors.
            1. input_ids_all: Tensor. Each sublist contains numerical values,
                i.e. token ids, corresponding to the tokens in the input 
                text data.
            2. input_mask_all: Tensor. Each sublist contains the attention
                mask of the input token id list, 1 for input tokens and 0 for
                padded tokens, so that padded tokens are not attended to.
            3. trailing_token_mask_all: Tensor. Each sublist is
                a boolean list, True for the first word piece of each
                original word, False for the trailing word pieces,
                e.g. "##ize". This mask is useful for removing the
                predictions on trailing word pieces, so that each
                original word in the input text has a unique predicted
                label.
            4. label_ids_all: Tensor, each sublist contains token labels of
                a input sentence/paragraph, if labels is provided. If the
                `labels` argument is not provided, it will not return this tensor.
    """

    train_df, test_df = load_train_test_dfs(
        local_cache_path=local_path,
        test_fraction=test_fraction,
        random_seed=random_seed
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

    processor = TokenClassificationProcessor(
        model_name=model_name,
        to_lower=to_lower,
        cache_dir=cache_dir
    )

    label_map = TokenClassificationProcessor.create_label_map(
        label_lists=train_df['labels'],
        trailing_piece_tag=trailing_piece_tag
    )

    train_dataset = processor.preprocess_for_bert(
        text=train_df['sentence'],
        max_len=max_len,
        labels=train_df['labels'],
        label_map=label_map,
        trailing_piece_tag=trailing_piece_tag
    )

    test_dataset = processor.preprocess_for_bert(
        text=test_df['sentence'],
        max_len=max_len,
        labels=test_df['labels'],
        label_map=label_map,
        trailing_piece_tag=trailing_piece_tag
    )

    train_dataloader = processor.create_dataloader_from_dataset(
        train_dataset,
        shuffle=True,
        batch_size=batch_size,
        num_gpus=num_gpus,
        distributed=False
    )

    test_dataloader = processor.create_dataloader_from_dataset(
        test_dataset,
        shuffle=False,
        batch_size=batch_size,
        num_gpus=num_gpus,
        distributed=False
    )

    return (train_dataloader, test_dataloader, label_map, test_dataset)
