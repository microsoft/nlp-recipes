# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Utility functions for loading the Cross-Lingual NLI Corpus (XNLI) as a PyTorch Dataset."""

import numpy as np
import torch
from utils_nlp.models.bert.common import Language, Tokenizer
from torch.utils import data
from utils_nlp.dataset.xnli import load_pandas_df
from sklearn.preprocessing import LabelEncoder

MAX_SEQ_LENGTH = 128
TEXT_COL = "text"
LABEL_COL = "label"
DATA_PERCENT_USED = 1.0
TRAIN_FILE_SPLIT = "train"
TEST_FILE_SPLIT = "test"
VALIDATION_FILE_SPLIT = "dev"
CACHE_DIR = "./"
LANGUAGE_ENGLISH = "en"
TO_LOWER_CASE = False
TOK_ENGLISH = Language.ENGLISH
VALID_FILE_SPLIT = [TRAIN_FILE_SPLIT, VALIDATION_FILE_SPLIT, TEST_FILE_SPLIT]


def _load_pandas_df(cache_dir, file_split, language, data_percent_used):
    df = load_pandas_df(local_cache_path=cache_dir, file_split=file_split, language=language)
    data_used_count = round(data_percent_used * df.shape[0])
    df = df.loc[:data_used_count]
    return df


def _tokenize(tok_language, to_lowercase, cache_dir, df):
    print("Create a tokenizer...")
    tokenizer = Tokenizer(language=tok_language, to_lower=to_lowercase, cache_dir=cache_dir)
    tokens = tokenizer.tokenize(df[TEXT_COL])

    print("Tokenize and preprocess text...")
    # tokenize
    token_ids, input_mask, token_type_ids = tokenizer.preprocess_classification_tokens(
        tokens, max_len=MAX_SEQ_LENGTH
    )
    return token_ids, input_mask, token_type_ids


def _fit_train_labels(df):
    label_encoder = LabelEncoder()
    train_labels = label_encoder.fit_transform(df[LABEL_COL])
    train_labels = np.array(train_labels)
    return label_encoder, train_labels


class XnliDataset(data.Dataset):
    def __init__(
        self,
        file_split=TRAIN_FILE_SPLIT,
        cache_dir=CACHE_DIR,
        language=LANGUAGE_ENGLISH,
        to_lowercase=TO_LOWER_CASE,
        tok_language=TOK_ENGLISH,
        data_percent_used=DATA_PERCENT_USED,
    ):
        """
            Load the dataset here
        Args:
            file_split (str, optional):The subset to load.
                One of: {"train", "dev", "test"}
                Defaults to "train".
            cache_dir (str, optional):Path to store the data.
                Defaults to "./".
            language(str):Language required to load which xnli file (eg - "en", "zh")
            to_lowercase(bool):flag to convert samples in dataset to lowercase
            tok_language(Language, optional): language (Language, optional): The pretrained model's
                language. Defaults to Language.ENGLISH.
            data_percent_used(float, optional): Data used to create Torch Dataset.
                Defaults to "1.0" which is 100% data
        """
        if file_split not in VALID_FILE_SPLIT:
            raise ValueError("The file split is not part of ", VALID_FILE_SPLIT)

        self.file_split = file_split
        self.cache_dir = cache_dir
        self.language = language
        self.to_lowercase = to_lowercase
        self.tok_language = tok_language
        self.data_percent_used = data_percent_used

        df = _load_pandas_df(self.cache_dir, self.file_split, self.language, self.data_percent_used)

        self.df = df

        token_ids, input_mask, token_type_ids = _tokenize(
            tok_language, to_lowercase, cache_dir, self.df
        )

        self.token_ids = token_ids
        self.input_mask = input_mask
        self.token_type_ids = token_type_ids

        if file_split == TRAIN_FILE_SPLIT:
            label_encoder, train_labels = _fit_train_labels(self.df)
            self.label_encoder = label_encoder
            self.labels = train_labels
        else:
            # use the label_encoder passed when you create the test/validate dataset
            self.labels = self.df[LABEL_COL]

    def __len__(self):
        """ Denotes the total number of samples """
        return len(self.df)

    def __getitem__(self, index):
        """ Generates one sample of data """
        token_ids = self.token_ids[index]
        input_mask = self.input_mask[index]
        token_type_ids = self.token_type_ids[index]
        labels = self.labels[index]

        return {
            "token_ids": torch.tensor(token_ids, dtype=torch.long),
            "input_mask": torch.tensor(input_mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "labels": labels,
        }
