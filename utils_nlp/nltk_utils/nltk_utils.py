import nltk


def nltk_tokenizer(snli_df):
    snli_df['sentence1_tokens'] = snli_df.apply(lambda row: nltk.word_tokenize(row['sentence1']), axis=1)
    snli_df['sentence2_tokens'] = snli_df.apply(lambda row: nltk.word_tokenize(row['sentence2']), axis=1)

    return snli_df
