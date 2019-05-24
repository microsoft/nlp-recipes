"""This script reuses some code from
https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/examples
/run_classifier.py"""

import pandas as pd
import csv
import sys
import random


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
        Converts input data into sentences and labels
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
        text_all = []
        labels_all = []
        for (i, sent) in enumerate(sentences):
            text_a = " ".join([s[0] for s in sent])
            label = [s[2] for s in sent]

            text_all.append(text_a)
            labels_all.append(label)

        return text_all, labels_all
