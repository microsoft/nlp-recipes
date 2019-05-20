"""This script reuses some code from
https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/examples
/run_classifier.py"""

import pandas as pd
import csv
import sys
import random
from collections import namedtuple


## Previous version of BERTInputData using class
# class BERTInputData(object):
#     """A single training/test example."""
#
#     def __init__(self, guid, text_a, text_b=None, label=None):
#         """
#         Constructs an InputExample object.
#
#         Args:
#             guid (str): Unique id for the example.
#             text_a (str): The untokenized text of the first sequence.
#                 For single sequence tasks, only this sequence must be
#                 specified.
#             text_b (str, optional): The untokenized text of the second
#                 sequence. Only must be specified for sequence pair tasks.
#             label (str, optional): The label of the example. This should be
#             specified for train and dev examples, but not for test examples.
#         """
#         self.text_a = text_a
#         self.text_b = text_b
#         self.label = label

## New version of BERTInputData using namedtuple
"""
A single BERT input data containing three fields:
    1. text_a: text of the first sentence,
    2. text_b: text of the second sentence, optional, required for
        two-sentence tasks.
    3. label: label, optional, required for training and validation data
"""
BertInputData = namedtuple(
    "BertInputData", ["text_a", "text_b", "label"], defaults=[None, None]
)


class DataProcessor(object):
    """Base class for data converters for classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `BertInputData`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `BertInputData`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, "utf-8") for cell in line)
                lines.append(line)
            return lines


class KaggleNERProcessor(DataProcessor):
    """
    Data processor for the Kaggle NER dataset:
    https://www.kaggle.com/abhinavwalia95/entity-annotated-corpus
    """

    def __init__(self, data_dir, dev_percentage):
        """
        Initializes the data processor.

        Args:
            data_dir (str): Directory to read the dataset from.
            dev_percentage (float): Percentage of data used as dev/validation
                data.
        """
        super().__init__()
        self.data_dir = data_dir
        data = self._read_data(data_dir)

        unique_sentence_nums = data["Sentence #"].unique()
        random.shuffle(unique_sentence_nums)
        sentence_count = len(unique_sentence_nums)
        dev_sentence_count = round(sentence_count * dev_percentage)
        train_sentence_count = sentence_count - dev_sentence_count

        self.train_sentence_nums = unique_sentence_nums[:train_sentence_count]
        self.dev_sentence_nums = unique_sentence_nums[train_sentence_count:]

        self.tag_vals = list(set(data["Tag"].values))
        self.tag_vals.append("X")

    def get_train_examples(self):
        """
        Gets the training data.
        """
        data = self._read_data(self.data_dir)
        train_data = data.loc[
            data["Sentence #"].isin(self.train_sentence_nums)
        ].copy()

        return self._create_examples(train_data)

    def get_dev_examples(self):
        """
        Gets the dev/validation data.
        """
        data = self._read_data(self.data_dir)
        dev_data = data.loc[
            data["Sentence #"].isin(self.dev_sentence_nums)
        ].copy()

        return self._create_examples(dev_data)

    def get_labels(self):
        """Gets a list of unique labels in this dataset.

        Returns:
            list: A list of unique labels in the dataset.
        """
        return self.tag_vals

    def _read_data(self, data_dir):
        return pd.read_csv(data_dir, encoding="latin1").fillna(method="ffill")

    def _create_examples(self, data):
        """
        Converts input data into BertInputData type.
        """
        agg_func = lambda s: [
            (w, p, t)
            for w, p, t in zip(
                s["Word"].values.tolist(),
                s["POS"].values.tolist(),
                s["Tag"].values.tolist(),
            )
        ]
        data_grouped = data.groupby("Sentence #").apply(agg_func)
        sentences = [s for s in data_grouped]
        examples = []
        for (i, sent) in enumerate(sentences):
            text_a = " ".join([s[0] for s in sent])
            label = [s[2] for s in sent]
            examples.append(
                BertInputData(text_a=text_a, text_b=None, label=label)
            )

        return examples
