# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pandas as pd
import spacy
import nltk
from nltk.corpus import stopwords


def to_lowercase(df):
    """Transform all strings in the dataframe to lowercase 

    Args:
        df (pandas dataframe): Raw dataframe with some text columns.

    Returns:
        pandas dataframe: Dataframe with lowercase standardization.
    """
    return df.applymap(lambda s: s.lower() if type(s) == str else s)


def to_spacy_tokens(df):
    """Tokenize using spaCy, defaulting to the spaCy en_core_web_sm model

    Args:
        df (pandas dataframe): Dataframe with columns labeled 'sentence1' and 'sentence2' to tokenize.

    Returns:
        pandas dataframe: Dataframe with new columns labeled 'sentence1_tokens' and 'sentence2_tokens', each containing 
                            a list of tokens for their respective sentences.
    """
    nlp = spacy.load("en_core_web_sm")
    text_df = df[["sentence1", "sentence2"]]
    nlp_df = text_df.applymap(lambda x: nlp(x))
    tok_df = nlp_df.applymap(lambda doc: [token.text for token in doc])
    tok_df.columns = ["sentence1_tokens", "sentence2_tokens"]
    tokenized = pd.concat([df, tok_df], axis=1)
    return tokenized


def rm_spacy_stopwords(df, custom_stopwords=[]):
    """Tokenize using spaCy AND remove stopwords, defaulting to the spaCy en_core_web_sm model

    Args:
        df (pandas dataframe): Dataframe with columns labeled 'sentence1' and 'sentence2' to tokenize.
        custom_stopwords (list of str, optional): List of custom stopwords to register with the spaCy model.

    Returns:
        pandas dataframe: Dataframe with new columns labeled 'sentence1_tokens_stop' and 'sentence2_tokens_stop', each 
                            containing a list of tokens for their respective sentences.
    """
    nlp = spacy.load("en_core_web_sm")
    if len(custom_stopwords) > 0:
        for csw in custom_stopwords:
            nlp.vocab[csw].is_stop = True
    text_df = df[["sentence1", "sentence2"]]
    nlp_df = text_df.applymap(lambda x: nlp(x))
    tok_df = nlp_df.applymap(
        lambda doc: [token.text for token in doc if not token.is_stop]
    )
    tok_df.columns = ["sentence1_tokens_stop", "sentence2_tokens_stop"]
    tokenized = pd.concat([df, tok_df], axis=1)
    return tokenized


def to_nltk_tokens(df):
    """
    This function converts a sentence to word tokens using nltk.
    It adds two new columns sentence1_tokens and sentence2_tokens to the input pandas dataframe
    Args:
        snli_df: pandas dataframe

    Returns:
        pandas dataframe with columns ['score','sentence1', 'sentence2', 'sentence1_tokens', 'sentence2_tokens']
    """
    df["sentence1_tokens"] = df.apply(
        lambda row: nltk.word_tokenize(row["sentence1"]), axis=1
    )
    df["sentence2_tokens"] = df.apply(
        lambda row: nltk.word_tokenize(row["sentence2"]), axis=1
    )

    return df


def nltk_remove_stop_words(snli_df):
    """
        This function removes stop words from a sentence using nltk.
        It adds two new columns sentence1_tokens_stop and sentence2_tokens_stop to the input pandas dataframe
    Args:
        snli_df: pandas dataframe

    Returns:
        pandas dataframe with columns ['score','sentence1', 'sentence2', 'sentence1_tokens', 'sentence2_tokens']
    """
    if not {"sentence1_tokens", "sentence2_tokens"}.issubset(snli_df.columns):
        snli_df = nltk_tokenizer(snli_df)

    stop_words = tuple(stopwords.words("english"))

    snli_df["sentence1_tokens_stop"] = [
        [word for word in row if word not in stop_words]
        for row in snli_df["sentence1_tokens"]
    ]
    snli_df["sentence2_tokens_stop"] = [
        [word for word in row if word not in stop_words]
        for row in snli_df["sentence2_tokens"]
    ]

    return snli_df
