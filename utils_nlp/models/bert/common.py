# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.


# This script reuses some code from
# https://github.com/huggingface/transformers/blob/master/examples
# /run_glue.py

import csv
import linecache
import subprocess
import warnings
from collections.abc import Iterable
from enum import Enum

import torch
from pytorch_pretrained_bert.tokenization import BertTokenizer
from torch.utils.data import (
    DataLoader,
    Dataset,
    RandomSampler,
    SequentialSampler,
    TensorDataset,
    ConcatDataset,
)
from tqdm import tqdm

# Max supported sequence length
BERT_MAX_LEN = 512


class Language(str, Enum):
    """An enumeration of the supported pretrained models and languages."""

    ENGLISH: str = "bert-base-uncased"
    ENGLISHCASED: str = "bert-base-cased"
    ENGLISHLARGE: str = "bert-large-uncased"
    ENGLISHLARGECASED: str = "bert-large-cased"
    ENGLISHLARGEWWM: str = "bert-large-uncased-whole-word-masking"
    ENGLISHLARGECASEDWWM: str = "bert-large-cased-whole-word-masking"
    CHINESE: str = "bert-base-chinese"
    MULTILINGUAL: str = "bert-base-multilingual-cased"


class Tokenizer:
    def __init__(self, language=Language.ENGLISH, to_lower=False, cache_dir="."):
        """Initializes the underlying pretrained BERT tokenizer.

        Args:
            language (Language, optional): The pretrained model's language.
                                           Defaults to Language.ENGLISH.
            cache_dir (str, optional): Location of BERT's cache directory.
                Defaults to ".".
        """
        self.tokenizer = BertTokenizer.from_pretrained(
            language, do_lower_case=to_lower, cache_dir=cache_dir
        )
        self.language = language

    def tokenize(self, text):
        """Tokenizes a list of documents using a BERT tokenizer

        Args:
            text (list): List of strings (one sequence) or
                tuples (two sequences).

        Returns:
            [list]: List of lists. Each sublist contains WordPiece tokens
                of the input sequence(s).
        """
        if isinstance(text[0], str):
            return [self.tokenizer.tokenize(x) for x in tqdm(text)]
        else:
            return [[self.tokenizer.tokenize(x) for x in sentences] for sentences in tqdm(text)]

    def _truncate_seq_pair(self, tokens_a, tokens_b, max_length):
        """Truncates a sequence pair in place to the maximum length."""
        # This is a simple heuristic which will always truncate the longer
        # sequence one token at a time. This makes more sense than
        # truncating an equal percent of tokens from each, since if one
        # sequence is very short then each token that's truncated likely
        # contains more information than a longer sequence.

        if not tokens_b:
            max_length += 1

        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()

        tokens_a.append("[SEP]")

        if tokens_b:
            tokens_b.append("[SEP]")

        return [tokens_a, tokens_b]

    def preprocess_classification_tokens(self, tokens, max_len=BERT_MAX_LEN):
        """Preprocessing of input tokens:
            - add BERT sentence markers ([CLS] and [SEP])
            - map tokens to token indices in the BERT vocabulary
            - pad and truncate sequences
            - create an input_mask
            - create token type ids, aka. segment ids

        Args:
            tokens (list): List of token lists to preprocess.
            max_len (int, optional): Maximum number of tokens
                            (documents will be truncated or padded).
                            Defaults to 512.
        Returns:
            tuple: A tuple containing the following three lists
                list of preprocesssed token lists
                list of input mask lists
                list of token type id lists
        """
        if max_len > BERT_MAX_LEN:
            print("setting max_len to max allowed tokens: {}".format(BERT_MAX_LEN))
            max_len = BERT_MAX_LEN

        if isinstance(tokens[0][0], str):
            tokens = [x[0 : max_len - 2] + ["[SEP]"] for x in tokens]
            token_type_ids = None
        else:
            # get tokens for each sentence [[t00, t01, ...] [t10, t11,... ]]
            tokens = [
                self._truncate_seq_pair(sentence[0], sentence[1], max_len - 3)
                for sentence in tokens
            ]

            # construct token_type_ids
            # [[0, 0, 0, 0, ... 0, 1, 1, 1, ... 1], [0, 0, 0, ..., 1, 1, ]
            token_type_ids = [
                [[i] * len(sentence) for i, sentence in enumerate(example)] for example in tokens
            ]
            # merge sentences
            tokens = [[token for sentence in example for token in sentence] for example in tokens]
            # prefix with [0] for [CLS]
            token_type_ids = [
                [0] + [i for sentence in example for i in sentence] for example in token_type_ids
            ]
            # pad sequence
            token_type_ids = [x + [0] * (max_len - len(x)) for x in token_type_ids]

        tokens = [["[CLS]"] + x for x in tokens]
        # convert tokens to indices
        tokens = [self.tokenizer.convert_tokens_to_ids(x) for x in tokens]
        # pad sequence
        tokens = [x + [0] * (max_len - len(x)) for x in tokens]
        # create input mask
        input_mask = [[min(1, x) for x in y] for y in tokens]
        return tokens, input_mask, token_type_ids

    def preprocess_encoder_tokens(self, tokens, max_len=BERT_MAX_LEN):
        """Preprocessing of input tokens:
            - add BERT sentence markers ([CLS] and [SEP])
            - map tokens to token indices in the BERT vocabulary
            - pad and truncate sequences
            - create an input_mask
            - create token type ids, aka. segment ids

        Args:
            tokens (list): List of token lists to preprocess.
            max_len (int, optional): Maximum number of tokens
                            (documents will be truncated or padded).
                            Defaults to 512.
        Returns:
            tuple: A tuple containing the following four lists
                list of preprocesssed token lists
                list of input id lists
                list of input mask lists
                list of token type id lists
        """
        if max_len > BERT_MAX_LEN:
            print("setting max_len to max allowed tokens: {}".format(BERT_MAX_LEN))
            max_len = BERT_MAX_LEN

        if isinstance(tokens[0][0], str):
            tokens = [x[0 : max_len - 2] + ["[SEP]"] for x in tokens]
            token_type_ids = None
        else:
            # get tokens for each sentence [[t00, t01, ...] [t10, t11,... ]]
            tokens = [
                self._truncate_seq_pair(sentence[0], sentence[1], max_len - 3)
                for sentence in tokens
            ]

            # construct token_type_ids
            # [[0, 0, 0, 0, ... 0, 1, 1, 1, ... 1], [0, 0, 0, ..., 1, 1, ]
            token_type_ids = [
                [[i] * len(sentence) for i, sentence in enumerate(example)] for example in tokens
            ]
            # merge sentences
            tokens = [[token for sentence in example for token in sentence] for example in tokens]
            # prefix with [0] for [CLS]
            token_type_ids = [
                [0] + [i for sentence in example for i in sentence] for example in token_type_ids
            ]
            # pad sequence
            token_type_ids = [x + [0] * (max_len - len(x)) for x in token_type_ids]

        tokens = [["[CLS]"] + x for x in tokens]
        # convert tokens to indices
        input_ids = [self.tokenizer.convert_tokens_to_ids(x) for x in tokens]
        # pad sequence
        input_ids = [x + [0] * (max_len - len(x)) for x in input_ids]
        # create input mask
        input_mask = [[min(1, x) for x in y] for y in input_ids]
        return tokens, input_ids, input_mask, token_type_ids

    def tokenize_ner(
        self, text, max_len=BERT_MAX_LEN, labels=None, label_map=None, trailing_piece_tag="X"
    ):
        """
        Tokenize and preprocesses input word lists, involving the following steps
            0. WordPiece tokenization.
            1. Convert string tokens to token ids.
            2. Convert input labels to label ids, if labels and label_map are
                provided.
            3. If a word is tokenized into multiple pieces of tokens by the
                WordPiece tokenizer, label the extra tokens with
                trailing_piece_tag.
            4. Pad or truncate input text according to max_seq_length
            5. Create input_mask for masking out padded tokens.

        Args:
            text (list): List of lists. Each sublist is a list of words in an
                input sentence.
            max_len (int, optional): Maximum length of the list of
                tokens. Lists longer than this are truncated and shorter
                ones are padded with "O"s. Default value is BERT_MAX_LEN=512.
            labels (list, optional): List of word label lists. Each sublist
                contains labels corresponding to the input word list. The lengths
                of the label list and word list must be the same. Default
                value is None.
            label_map (dict, optional): Dictionary for mapping original token
                labels (which may be string type) to integers. Default value
                is None.
            trailing_piece_tag (str, optional): Tag used to label trailing
                word pieces. For example, "criticize" is broken into "critic"
                and "##ize", "critic" preserves its original label and "##ize"
                is labeled as trailing_piece_tag. Default value is "X".

        Returns:
            tuple: A tuple containing the following four lists.
                1. input_ids_all: List of lists. Each sublist contains
                    numerical values, i.e. token ids, corresponding to the
                    tokens in the input text data.
                2. input_mask_all: List of lists. Each sublist
                    contains the attention mask of the input token id list,
                    1 for input tokens and 0 for padded tokens, so that
                    padded tokens are not attended to.
                3. trailing_token_mask: List of lists. Each sublist is
                    a boolean list, True for the first word piece of each
                    original word, False for the trailing word pieces,
                    e.g. "##ize". This mask is useful for removing the
                    predictions on trailing word pieces, so that each
                    original word in the input text has a unique predicted
                    label.
                4. label_ids_all: List of lists of numerical labels,
                    each sublist contains token labels of a input
                    sentence/paragraph, if labels is provided. If the `labels`
                    argument is not provided, the value of this is None.
        """

        def _is_iterable_but_not_string(obj):
            return isinstance(obj, Iterable) and not isinstance(obj, str)

        if max_len > BERT_MAX_LEN:
            warnings.warn("setting max_len to max allowed tokens: {}".format(BERT_MAX_LEN))
            max_len = BERT_MAX_LEN

        if not _is_iterable_but_not_string(text):
            # The input text must be an non-string Iterable
            raise ValueError("Input text must be an iterable and not a string.")
        else:
            # If the input text is a single list of words, convert it to
            # list of lists for later iteration
            if not _is_iterable_but_not_string(text[0]):
                text = [text]
        if labels is not None:
            if not _is_iterable_but_not_string(labels):
                raise ValueError("labels must be an iterable and not a string.")
            else:
                if not _is_iterable_but_not_string(labels[0]):
                    labels = [labels]

        label_available = True
        if labels is None:
            label_available = False
            # create an artificial label list for creating trailing token mask
            labels = [["O"] * len(t) for t in text]

        input_ids_all = []
        input_mask_all = []
        label_ids_all = []
        trailing_token_mask_all = []
        for t, t_labels in zip(text, labels):

            if len(t) != len(t_labels):
                raise ValueError(
                    "The number of words is {0}, but the number of labels is {1}.".format(
                        len(t), len(t_labels)
                    )
                )

            new_labels = []
            new_tokens = []
            if label_available:
                for word, tag in zip(t, t_labels):
                    sub_words = self.tokenizer.tokenize(word)
                    for count, sub_word in enumerate(sub_words):
                        if count > 0:
                            tag = trailing_piece_tag
                        new_labels.append(tag)
                        new_tokens.append(sub_word)
            else:
                for word in t:
                    sub_words = self.tokenizer.tokenize(word)
                    for count, sub_word in enumerate(sub_words):
                        if count > 0:
                            tag = trailing_piece_tag
                        else:
                            tag = "O"
                        new_labels.append(tag)
                        new_tokens.append(sub_word)

            if len(new_tokens) > max_len:
                new_tokens = new_tokens[:max_len]
                new_labels = new_labels[:max_len]
            input_ids = self.tokenizer.convert_tokens_to_ids(new_tokens)

            # The mask has 1 for real tokens and 0 for padding tokens.
            # Only real tokens are attended to.
            input_mask = [1.0] * len(input_ids)

            # Zero-pad up to the max sequence length.
            padding = [0.0] * (max_len - len(input_ids))
            label_padding = ["O"] * (max_len - len(input_ids))

            input_ids += padding
            input_mask += padding
            new_labels += label_padding

            trailing_token_mask_all.append(
                [True if label != trailing_piece_tag else False for label in new_labels]
            )

            if label_map:
                label_ids = [label_map[label] for label in new_labels]
            else:
                label_ids = new_labels

            input_ids_all.append(input_ids)
            input_mask_all.append(input_mask)
            label_ids_all.append(label_ids)

        if label_available:
            return (input_ids_all, input_mask_all, trailing_token_mask_all, label_ids_all)
        else:
            return input_ids_all, input_mask_all, trailing_token_mask_all, None


def create_data_loader(
    input_ids, input_mask, label_ids=None, sample_method="random", batch_size=32
):
    """
    Create a dataloader for sampling and serving data batches.

    Args:
        input_ids (list): List of lists. Each sublist contains numerical
            values, i.e. token ids, corresponding to the tokens in the input
            text data.
        input_mask (list): List of lists. Each sublist contains the attention
            mask of the input token id list, 1 for input tokens and 0 for
            padded tokens, so that padded tokens are not attended to.
        label_ids (list, optional): List of lists of numerical labels,
            each sublist contains token labels of a input
            sentence/paragraph. Default value is None.
        sample_method (str, optional): Order of data sampling. Accepted
            values are "random", "sequential". Default value is "random".
        batch_size (int, optional): Number of samples used in each training
            iteration. Default value is 32.

    Returns:
        DataLoader: A Pytorch Dataloader containing the input_ids tensor,
            input_mask tensor, and label_ids (if provided) tensor.

    """
    input_ids_tensor = torch.tensor(input_ids, dtype=torch.long)
    input_mask_tensor = torch.tensor(input_mask, dtype=torch.long)

    if label_ids:
        label_ids_tensor = torch.tensor(label_ids, dtype=torch.long)
        tensor_data = TensorDataset(input_ids_tensor, input_mask_tensor, label_ids_tensor)
    else:
        tensor_data = TensorDataset(input_ids_tensor, input_mask_tensor)

    if sample_method == "random":
        sampler = RandomSampler(tensor_data)
    elif sample_method == "sequential":
        sampler = SequentialSampler(tensor_data)
    else:
        raise ValueError(
            "Invalid sample_method value, accepted values are: " "random and sequential."
        )

    dataloader = DataLoader(tensor_data, sampler=sampler, batch_size=batch_size)

    return dataloader


class TextDataset(Dataset):
    """
    Characterizes a dataset for PyTorch which can be used to load a file containing multiple rows
    where each row is a training example. The format of each line in the file is assumed to be
    tokens, mask and label.
    """

    def __init__(self, filename):
        """
        Initialization. We set the filename and number of lines in the file.
        Args:
            filename(str): Name of the file.
        """
        self._filename = filename
        self._total_data = (
            int(subprocess.check_output("wc -l " + filename, shell=True).split()[0]) - 1
        )

    def __len__(self):
        """Denotes the total number of samples in the file."""
        return self._total_data

    @staticmethod
    def _cast(row):
        return [int(x.strip()) for x in row]

    def __getitem__(self, index):
        """
        Generates one sample of data. We assume that the last column is label here. We use
        linecache to load files lazily.

        Args:
            index(int): Index of the test case.

        Returns(list, list, int): Returns the tokens, mask and label for a single item.

        """
        line = linecache.getline(self._filename, index + 1)
        row = next(csv.reader([line]))

        tokens = self._cast(row[0][1:-1].split(","))
        mask = self._cast(row[1][1:-1].split(","))

        data = {
            "token_ids": torch.tensor(tokens, dtype=torch.long),
            "input_mask": torch.tensor(mask, dtype=torch.long),
            "labels": torch.tensor(int(row[2]), dtype=torch.long),
        }

        return data


def get_dataset_multiple_files(files):
    """ Get dataset from multiple files

    Args:
        files(list): List of paths to the files.

    Returns:

        torch.utils.data.Dataset : A combined dataset of all files in the directory.

    """
    datasets = [TextDataset(x) for x in files]
    return ConcatDataset(datasets)
