import pandas as pd
import csv, sys, unicode, random


class InputExample(object):
    """A single training/test example."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For
            single sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second
            sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
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
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines


class KaggleNERProcessor(DataProcessor):
    """
        Data processor for the Kaggle NER dataset:
        https://www.kaggle.com/abhinavwalia95/entity-annotated-corpus
    """
    def __init__(self, data_dir, dev_percentage):
        super().__init__()
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

    def get_train_examples(self, data_dir):
        data = self._read_data(data_dir)
        train_data = data.loc[
            data["Sentence #"].isin(self.train_sentence_nums)].copy()

        return self._create_examples(train_data, "train")

    def get_dev_examples(self, data_dir):
        data = self._read_data(data_dir)
        dev_data = data.loc[
            data["Sentence #"].isin(self.dev_sentence_nums)].copy()

        return self._create_examples(dev_data, "dev")

    def get_labels(self):
        """Gets the list of labels for this data set."""
        return self.tag_vals

    @staticmethod
    def _read_data(data_dir):
        return pd.read_csv(data_dir, encoding="latin1").fillna(method="ffill")

    @staticmethod
    def _create_examples(data, set_type):
        agg_func = lambda s: [(w, p, t) for w, p, t in
                              zip(s["Word"].values.tolist(),
                                  s["POS"].values.tolist(),
                                  s["Tag"].values.tolist())]
        data_grouped = data.groupby("Sentence #").apply(agg_func)
        sentences = [s for s in data_grouped]
        examples = []
        for (i, sent) in enumerate(sentences):
            guid = "%s-%s" % (set_type, i)
            text_a = " ".join([s[0] for s in sent])
            label = [s[2] for s in sent]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None,
                             label=label))
        return examples