# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import random

from .url_utils import maybe_download

URL = (
    "https://raw.githubusercontent.com/juand-r/entity-recognition-datasets"
    "/master/data/wikigold/CONLL-format/data/wikigold.conll.txt"
)


def download(dir_path):
    """Download the wikigold data file to dir_path if it doesn't exist yet."""
    file_name = URL.split("/")[-1]
    maybe_download(URL, file_name, dir_path)


def read_data(data_file):
    """
    Read the wikigold dataset into a string of text.

    Args:
        data_file (str): data file path, including the file name.

    Returns:
        str: One string containing the wikigold dataset.
    """
    with open(data_file, "r", encoding="utf8") as file:
        text = file.read()

    return text


def get_train_test_data(text, test_percentage=0.5, random_seed=None):
    """
    Get the training and testing data based on test_percentage.

    Args:
        text (str): One string containing the wikigold dataset.
        test_percentage (float, optional): Percentage of data ot use for
            testing. Since this is a small dataset, the default testing
            percentage is set to 0.5
        random_seed (float, optional): Random seed used to shuffle the data.

    Returns:
        tuple: A tuple containing four lists:
            train_sentence_list: List of training sentence strings.
            train_labels_list: List of lists. Each sublist contains the
                entity labels of the words in the training sentence.
            test_sentence_list: List of testing sentence strings.
            test_labels_list: List of lists. Each sublist contains the
                entity labels of the word in the testing sentence.
    """
    # Input data are separated by empty lines
    text_split = text.split("\n\n")
    # Remove empty line at EOF
    text_split = text_split[:-1]

    if random_seed:
        random.seed(random_seed)
    random.shuffle(text_split)

    sentence_count = len(text_split)
    test_sentence_count = round(sentence_count * test_percentage)
    test_text_split = text_split[:test_sentence_count]
    train_text_split = text_split[test_sentence_count:]

    def _get_sentence_and_labels(text_list, data_type):
        max_seq_len = 0
        sentence_list = []
        labels_list = []
        for s in text_list:
            # split each sentence string into "word label" pairs
            s_split = s.split("\n")
            # split "word label" pairs
            s_split_split = [t.split() for t in s_split]
            sentence_list.append(" ".join([t[0] for t in s_split_split]))
            labels_list.append([t[1] for t in s_split_split])
            if len(s_split_split) > max_seq_len:
                max_seq_len = len(s_split_split)
        print(
            "Maximum sequence length in {0} data is: {1}".format(
                data_type, max_seq_len
            )
        )
        return sentence_list, labels_list

    train_sentence_list, train_labels_list = _get_sentence_and_labels(
        train_text_split, "training"
    )

    test_sentence_list, test_labels_list = _get_sentence_and_labels(
        test_text_split, "testing"
    )

    return (
        train_sentence_list,
        train_labels_list,
        test_sentence_list,
        test_labels_list,
    )


def get_unique_labels():
    """Get the unique labels in the wikigold dataset."""
    return ["O", "I-LOC", "I-MISC", "I-PER", "I-ORG", "X"]
