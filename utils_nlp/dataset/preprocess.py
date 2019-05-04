# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pandas as pd
import spacy
import nltk
from nltk.corpus import stopwords


def to_lowercase(df):
    """
	This function transforms all strings in the dataframe to lowercase 

	Args:
		df (pd.DataFrame): Raw dataframe with some text columns.

	Returns:
		pd.DataFrame: Dataframe with lowercase standardization.
	"""
    return df.applymap(lambda s: s.lower() if type(s) == str else s)


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
    tok_df.columns = token_cols
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
		sentence_cols (list, optional): Column names for the raw sentence pairs.
		token_cols (list, optional): Column names for the tokenized sentences.
	
	Returns:
		pd.DataFrame: Dataframe with new columns token_cols, each containing a 
			list of tokens for their respective sentences.
	"""
    nltk.download("punkt")
    df[token_cols[0]] = df.apply(
        lambda row: nltk.word_tokenize(row[sentence_cols[0]]), axis=1
    )
    df[token_cols[1]] = df.apply(
        lambda row: nltk.word_tokenize(row[sentence_cols[1]]), axis=1
    )

    return df


def rm_nltk_stopwords(
    df,
    token_cols=["sentence1_tokens", "sentence2_tokens"],
    stop_cols=[
        "sentence1_tokens_rm_stopwords",
        "sentence2_tokens_rm_stopwords",
    ],
):
    """
	This function removes stop words from a sentence using nltk.
	
	Args:
		df (pd.DataFrame): Dataframe with columns sentence_cols to tokenize.
		token_cols (list, optional): Column names for the tokenized sentence 
			pairs.
		stop_cols (list, optional): Column names for the tokenized sentences 
			without stop words.
	
	Returns:
		pd.DataFrame: Dataframe with new columns stop_cols, each containing a 
			list of tokens for their respective sentences.
	"""
    if not set(tok_cols).issubset(df.columns):
        df = to_nltk_tokens(df)

    stop_words = tuple(stopwords.words("english"))

    df[stop_cols[0]] = [
        [word for word in row if word not in stop_words]
        for row in df[token_cols[0]]
    ]
    df[stop_cols[1]] = [
        [word for word in row if word not in stop_words]
        for row in df[token_cols[1]]
    ]

    return df
