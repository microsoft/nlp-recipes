# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import random
import os
import pandas as pd

from utils_nlp.dataset.url_utils import maybe_download
from utils_nlp.dataset.ner_utils import preprocess_conll

URL = (
    "https://raw.githubusercontent.com/juand-r/entity-recognition-datasets"
    "/master/data/wikigold/CONLL-format/data/wikigold.conll.txt"
)


def load_train_test_dfs(
    local_cache_path="./", test_percentage=0.5, random_seed=None
):
    """
    Get the training and testing data frames based on test_percentage.

    Args:
        local_cache_path (str): Path to store the data. If the data file
            doesn't exist in this path, it's downloaded.
        test_percentage (float, optional): Percentage of data ot use for
            testing. Since this is a small dataset, the default testing
            percentage is set to 0.5
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
    test_sentence_count = round(sentence_count * test_percentage)
    test_sentence_list = sentence_list[:test_sentence_count]
    test_labels_list = labels_list[:test_sentence_count]
    train_sentence_list = sentence_list[test_sentence_count:]
    train_labels_list = labels_list[test_sentence_count:]

    train_df = pd.DataFrame(
        {"sentence": train_sentence_list, "labels": train_labels_list}
    )

    test_df = pd.DataFrame(
        {"sentence": test_sentence_list, "labels": test_labels_list}
    )

    return (train_df, test_df)


def get_unique_labels():
    """Get the unique labels in the wikigold dataset."""
    return ["O", "I-LOC", "I-MISC", "I-PER", "I-ORG"]
