# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Common helper functions for preprocessing Named Entity Recognition (NER) datasets."""


def preprocess_conll(text, sep="\t"):
    """
    Converts data in CoNLL format to word and label lists.

    Args:
        text (str): Text string in conll format, e.g.
            "Amy B-PER
             ADAMS I-PER
             works O
             at O
             the O
             University B-ORG
             of I-ORG
             Minnesota I-ORG
             . O"
        sep (str, optional): Column separator
            Defaults to \t
    Returns:
        tuple:
            (list of word lists, list of token label lists)
    """
    text_list = text.split("\n\n")
    if text_list[-1] in (" ", ""):
        text_list = text_list[:-1]

    max_seq_len = 0
    sentence_list = []
    labels_list = []
    for s in text_list:
        # split each sentence string into "word label" pairs
        s_split = s.split("\n")
        # split "word label" pairs
        s_split_split = [t.split(sep) for t in s_split]
        sentence_list.append([t[0] for t in s_split_split if len(t) > 1])
        labels_list.append([t[1] for t in s_split_split if len(t) > 1])
        if len(s_split_split) > max_seq_len:
            max_seq_len = len(s_split_split)
    print("Maximum sequence length is: {0}".format(max_seq_len))
    return sentence_list, labels_list


def read_conll_file(file_path, sep="\t", encoding=None):
    """
    Reads a data file in CoNLL format and returns word and label lists.

    Args:
        file_path (str): Data file path.
        sep (str, optional): Column separator. Defaults to "\t".
        encoding (str): File encoding used when reading the file.
            Defaults to None.

    Returns:
        (list, list): A tuple of word and label lists (list of lists).
    """
    with open(file_path, encoding=encoding) as f:
        data = f.read()
    return preprocess_conll(data, sep=sep)
