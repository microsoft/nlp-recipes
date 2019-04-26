# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pandas as pd
import csv


def load_glove():
    # ToDo move this to Azure blob
    file_path = "../../../Pretrained Vectors/glove.840B.300d.txt"
    words = pd.read_csv(
        file_path, sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE
    )
    return words


def get_vector(words, word):
    """ Return vector for a given word
    Args:
        words: The loaded glove vectors.
        word: A word token

    Returns: A numpy matrix that is the vector representation of the word.
    """
    return words.loc[word.values]


if __name__ == "__main__":
    print(get_vector(load_glove(), "hello"))
