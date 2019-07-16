# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import argparse
import logging
import os

import pandas as pd

from utils_nlp.bert.common import Language, Tokenizer

LABEL_COL = "genre"
TEXT_COL = "sentence1"
LANGUAGE = Language.ENGLISH
TO_LOWER = True
MAX_LEN = 150

logger = logging.getLogger(__name__)


def tokenize(df):
    """Tokenize the text documents and convert them to lists of tokens using the BERT tokenizer.
    Args:
        df(pd.Dataframe): Dataframe with training or test samples

    Returns:

        list: List of lists of tokens for train set.

    """
    tokenizer = Tokenizer(
        LANGUAGE, to_lower=TO_LOWER)
    tokens = tokenizer.tokenize(list(df[TEXT_COL]))

    return tokens


def preprocess(tokens):
    """ Preprocess method that does the following,
            Convert the tokens into token indices corresponding to the BERT tokenizer's vocabulary
            Add the special tokens [CLS] and [SEP] to mark the beginning and end of a sentence
            Pad or truncate the token lists to the specified max length
            Return mask lists that indicate paddings' positions
            Return token type id lists that indicate which sentence the tokens belong to (not needed
            for one-sequence classification)

    Args:
        tokens(pd.Dataframe): Dataframe with tokens for train set.

    Returns:
        list: List of lists of tokens for train or test set with special tokens added.
        list: Input mask.
    """
    tokenizer = Tokenizer(
        LANGUAGE, to_lower=TO_LOWER)
    tokens, mask, _ = tokenizer.preprocess_classification_tokens(
        tokens, MAX_LEN
    )

    return tokens, mask


parser = argparse.ArgumentParser()
parser.add_argument("--input_data", type=str, help="input data")
parser.add_argument("--output_data", type=str, help="output data directory")
parser.add_argument("--output_filename", type=str, help="output file name")

args = parser.parse_args()
input_data = args.input_data
output_data = args.output_data

if output_data is not None:
    print(output_data)
    os.makedirs(output_data, exist_ok=True)
    logger.info("%s created" % output_data)

df = pd.read_csv(args.input_data)
tokens_array = tokenize(df)
tokens_array, mask_array = preprocess(tokens_array)

df['tokens'] = tokens_array
df['mask'] = mask_array

# Filter columns
cols = ['tokens', 'mask', 'label']
df = df[cols]
df.to_csv(os.path.join(args.output_data, "output_filename"))
logger.info("Completed")
