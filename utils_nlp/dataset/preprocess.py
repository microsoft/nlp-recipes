# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pandas as pd
import spacy
import nltk
from nltk.corpus import stopwords


def to_lowercase_all(df):
    """
    This function transforms all strings in the dataframe to lowercase

    Args:
        df (pd.DataFrame): Raw dataframe with some text columns.

    Returns:
        pd.DataFrame: Dataframe with lowercase standardization.
    """
    return df.applymap(lambda s: s.lower() if type(s) == str else s)


def to_lowercase(df, column_names=[]):
    """
    This function transforms strings of the column names in the dataframe passed to lowercase

    Args:
        df (pd.DataFrame): Raw dataframe with some text columns.
        column_names(list, optional): column names to be changed to lowercase.

    Returns:
        pd.DataFrame: Dataframe with columns with lowercase standardization.
    """
    if not column_names:
        to_lowercase_all(df)
    else:
        df[column_names] = df[column_names].applymap(
            lambda s: s.lower() if type(s) == str else s
        )
        return df


def to_spacy_tokens(
    df,
    sentence_cols=["sentence1", "sentence2"],
    token_cols=["sentence1_tokens", "sentence2_tokens"],
):
    """
	This function tokenizes the sentence pairs using spaCy, defaulting to the 
	spaCy en_core_web_sm model
	
	Args:
		df (pd.DataFrame): Dataframe with columns sentence_cols to tokenize.
		sentence_cols (list, optional): Column names of the raw sentence pairs.
		token_cols (list, optional): Column names for the tokenized sentences.
	
	Returns:
		pd.DataFrame: Dataframe with new columns token_cols, each containing 
							a list of tokens for their respective sentences.
	"""
    nlp = spacy.load("en_core_web_sm")
    text_df = df[sentence_cols]
    nlp_df = text_df.applymap(lambda x: nlp(x))
    tok_df = nlp_df.applymap(lambda doc: [token.text for token in doc])
    tok_df.columns = token_cols
    tokenized = pd.concat([df, tok_df], axis=1)
    return tokenized


def rm_spacy_stopwords(
    df,
    sentence_cols=["sentence1", "sentence2"],
    stop_cols=[
        "sentence1_tokens_rm_stopwords",
        "sentence2_tokens_rm_stopwords",
    ],
    custom_stopwords=[],
):
    """
	This function tokenizes the sentence pairs using spaCy and remove stopwords, 
	defaulting to the spaCy en_core_web_sm model
	
	Args:
		df (pd.DataFrame): Dataframe with columns sentence_cols to tokenize.
		sentence_cols (list, optional): Column names for the raw sentence pairs.
		stop_cols (list, optional): Column names for the tokenized sentences 
			without stop words.
		custom_stopwords (list of str, optional): List of custom stopwords to 
			register with the spaCy model.
	
	Returns:
		pd.DataFrame: Dataframe with new columns stop_cols, each containing a 
			list of tokens for their respective sentences.
	"""
    nlp = spacy.load("en_core_web_sm")
    if len(custom_stopwords) > 0:
        for csw in custom_stopwords:
            nlp.vocab[csw].is_stop = True
    text_df = df[sentence_cols]
    nlp_df = text_df.applymap(lambda x: nlp(x))
    tok_df = nlp_df.applymap(
        lambda doc: [token.text for token in doc if not token.is_stop]
    )
    tok_df.columns = stop_cols
    tokenized = pd.concat([df, tok_df], axis=1)
    return tokenized


def to_nltk_tokens(
    df,
    sentence_cols=["sentence1", "sentence2"],
    token_cols=["sentence1_tokens", "sentence2_tokens"],
):
    """
    This function converts a sentence to word tokens using nltk.

    Args:
        df (pd.DataFrame): Dataframe with columns sentence_cols to tokenize.
        sentence_cols (list, optional): Column names for the raw sentences.
        token_cols (list, optional): Column names for the tokenized sentences.

    Returns:
    pd.DataFrame: Dataframe with new columns token_cols, each containing a
    list of tokens for their respective sentences.
    """

    nltk.download("punkt")
    df[token_cols] = df[sentence_cols].applymap(
        lambda sentence: nltk.word_tokenize(sentence)
    )
    pd.concat([df[sentence_cols], df[token_cols]], axis=1)
    return df


def rm_nltk_stopwords(
    df,
    sentence_cols=["sentence1", "sentence2"],
    stop_cols=[
        "sentence1_tokens_rm_stopwords",
        "sentence2_tokens_rm_stopwords",
    ],
):
    """
    This function removes stop words from a sentence using nltk.

    Args:
        df (pd.DataFrame): Dataframe with columns sentence_cols to tokenize.
        sentence_cols (list, optional): Column names for the raw entences.
        stop_cols (list, optional): Column names for the tokenized sentences
            without stop words.

    Returns:
        pd.DataFrame: Dataframe with new columns stop_cols, each containing a
        list of tokens for their respective sentences.
    """

    stop_words = tuple(stopwords.words("english"))

    df[stop_cols] = (
        df[sentence_cols]
        .applymap(lambda sentence: nltk.word_tokenize(sentence))
        .applymap(lambda l: [word for word in l if word not in stop_words])
    )

    return df
