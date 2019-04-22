import nltk
from nltk.corpus import stopwords


def nltk_tokenizer(snli_df):
    """
    This function converts a sentence to word tokens using nltk.
    It adds two new columns sentence1_tokens and sentence2_tokens to the input pandas dataframe
    Args:
        snli_df: pandas dataframe

    Returns:
        pandas dataframe with columns ['score','sentence1', 'sentence2', 'sentence1_tokens', 'sentence2_tokens']
    """
    snli_df["sentence1_tokens"] = snli_df.apply(
        lambda row: nltk.word_tokenize(row["sentence1"]), axis=1
    )
    snli_df["sentence2_tokens"] = snli_df.apply(
        lambda row: nltk.word_tokenize(row["sentence2"]), axis=1
    )

    return snli_df


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
