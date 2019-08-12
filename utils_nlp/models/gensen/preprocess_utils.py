# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import shutil
from utils_nlp.models.gensen import SNLI_CLEAN_PATH


def _preprocess(split_map, data_path, column_names):
    """
    Method to save the tokens for each split in a snli_1.0_{split}.txt.clean file,
    with the sentence pairs and scores tab-separated and the tokens separated by a
    single space.

    Args:
        split_map(dict) : A dictionary containing train, test and dev
        tokenized dataframes.
        data_path(str): Path to the data folder.
        column_names(list): List of column names for the new columns created.

    """

    for file_split, df in split_map.items():
        base_txt_path = os.path.join(
            data_path, SNLI_CLEAN_PATH, "snli_1.0_{}.txt".format(file_split)
        )

        df[column_names[0]] = df["sentence1_tokens"].apply(
            lambda x: " " "".join(x)
        )
        df[column_names[1]] = df["sentence2_tokens"].apply(
            lambda x: " " "".join(x)
        )
        df[column_names[0]].to_csv(
            "{}.s1.tok".format(base_txt_path),
            sep=" ",
            header=False,
            index=False,
        )
        df[column_names[1]].to_csv(
            "{}.s2.tok".format(base_txt_path),
            sep=" ",
            header=False,
            index=False,
        )
        df[column_names[2]].to_csv(
            "{}.lab".format(base_txt_path), sep=" ", header=False, index=False
        )
        df_clean = df[column_names]
        df_clean.to_csv(
            "{}.clean".format(base_txt_path),
            sep="\t",
            header=False,
            index=False,
        )
        # remove rows with blank scores
        df_noblank = df_clean.loc[df_clean[column_names[2]] != "-"].copy()
        df_noblank.to_csv(
            "{}.clean.noblank".format(base_txt_path),
            sep="\t",
            header=False,
            index=False,
        )


def _split_and_cleanup(split_map, data_path):
    """
    Method that removes quotations from .tok files and saves the tokenized sentence
    and labels separately, in the form snli_1.0_{split}.txt.s1.tok or snli_1.0_{
    split}.txt.s2.tok or snli_1.0_{split}.txt.lab.

    Args:
        split_map(dict) : A dictionary containing train, test and dev
        tokenized dataframes.
        data_path(str): Path to the data folder.

    """

    for file_split in split_map.keys():

        s1_tok_path = os.path.join(
            data_path,
            SNLI_CLEAN_PATH,
            "snli_1.0_{}.txt.s1.tok".format(file_split),
        )
        s2_tok_path = os.path.join(
            data_path,
            SNLI_CLEAN_PATH,
            "snli_1.0_{}.txt.s2.tok".format(file_split),
        )
        with open(s1_tok_path, "r") as fin, open(
            "{}.tmp".format(s1_tok_path), "w"
        ) as tmp:
            for line in fin:
                s = line.replace('"', "")
                tmp.write(s)
        with open(s2_tok_path, "r") as fin, open(
            "{}.tmp".format(s2_tok_path), "w"
        ) as tmp:
            for line in fin:
                s = line.replace('"', "")
                tmp.write(s)
        shutil.move("{}.tmp".format(s1_tok_path), s1_tok_path)
        shutil.move("{}.tmp".format(s2_tok_path), s2_tok_path)


def gensen_preprocess(train_tok, dev_tok, test_tok, data_path):
    """
    Method to preprocess the train, validation and test datasets according to Gensen
    models requirements.

    Args:
        train_tok(pd.Dataframe): Tokenized training dataframe.
        dev_tok(pd.Dataframe): Tokenized validation dataframe.
        test_tok(pd.Dataframe): Tokenized test dataframe.
        data_path(str): Path to the data folder.

    Returns:
        str: Path to the processed dataset for GenSen.

    """

    split_map = {}

    if train_tok is not None:
        split_map["train"] = train_tok
    if dev_tok is not None:
        split_map["dev"] = dev_tok
    if test_tok is not None:
        split_map["test"] = test_tok

    column_names = ["s1.tok", "s2.tok", "score"]

    if not os.path.exists(os.path.join(data_path, SNLI_CLEAN_PATH)):
        os.makedirs(os.path.join(data_path, SNLI_CLEAN_PATH), exist_ok=True)

    _preprocess(split_map, data_path, column_names)
    _split_and_cleanup(split_map, data_path)

    return os.path.join(data_path, SNLI_CLEAN_PATH)
