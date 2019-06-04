# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import shutil

SPLIT_MAP = {}


def _preprocess(data_path):
    """
    Method to save the tokens for each split in a snli_1.0_{split}.txt.clean file,
    with the sentence pairs and scores tab-separated and the tokens separated by a
    single space.

    Args:
        data_path(str): Path to the data folder.

    """

    for file_split, df in SPLIT_MAP.items():
        base_txt_path = os.path.join(
            data_path, "clean/snli_1.0/snli_1.0_{}.txt".format(file_split)
        )

        df["s1.tok"] = df["sentence1_tokens"].apply(lambda x: " ".join(x))
        df["s2.tok"] = df["sentence2_tokens"].apply(lambda x: " ".join(x))
        df["s1.tok"].to_csv(
            "{}.s1.tok".format(base_txt_path), sep=" ", header=False, index=False
        )
        df["s2.tok"].to_csv(
            "{}.s2.tok".format(base_txt_path), sep=" ", header=False, index=False
        )
        df["score"].to_csv(
            "{}.lab".format(base_txt_path), sep=" ", header=False, index=False
        )
        df_clean = df[["s1.tok", "s2.tok", "score"]]
        df_clean.to_csv(
            "{}.clean".format(base_txt_path), sep="\t", header=False, index=False
        )
        # remove rows with blank scores
        df_noblank = df_clean.loc[df_clean["score"] != "-"].copy()
        print(base_txt_path)
        df_noblank.to_csv(
            "{}.clean.noblank".format(base_txt_path), sep="\t", header=False,
            index=False
        )


def _split_and_cleanup(data_path):
    """
    Method that removes quotations from .tok files and saves the tokenized sentence
    and labels separately, in the form snli_1.0_{split}.txt.s1.tok or snli_1.0_{
    split}.txt.s2.tok or snli_1.0_{split}.txt.lab.

    Args:
        data_path: Path to the data folder.

    """

    for file_split in SPLIT_MAP.keys():
        s1_tok_path = os.path.join(
            data_path, "clean/snli_1.0/snli_1.0_{}.txt.s1.tok".format(file_split)
        )
        s2_tok_path = os.path.join(
            data_path, "clean/snli_1.0/snli_1.0_{}.txt.s2.tok".format(file_split)
        )
        with open(s1_tok_path, "r") as fin, open("{}.tmp".format(s1_tok_path),
                                                 "w") as tmp:
            for line in fin:
                s = line.replace('"', "")
                tmp.write(s)
        with open(s2_tok_path, "r") as fin, open("{}.tmp".format(s2_tok_path),
                                                 "w") as tmp:
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

    """
    global SPLIT_MAP
    SPLIT_MAP = {}

    if train_tok is not None:
        SPLIT_MAP["train"] = train_tok
    if dev_tok is not None:
        SPLIT_MAP["dev"] = dev_tok
    if test_tok is not None:
        SPLIT_MAP["test"] = test_tok

    _preprocess(data_path)
    _split_and_cleanup(data_path)

    return os.path.join(data_path, "clean/snli_1.0")
