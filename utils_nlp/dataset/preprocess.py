import pandas as pd
import spacy

def to_lowercase(df):
	return df.applymap(lambda s: s.lower() if type(s) == str else s)

def to_spacy_tokens(df, custom_stop_words = []):
    nlp = spacy.load("en_core_web_sm")
    if len(custom_stop_words) > 0:
        for csw in custom_stop_words:
            nlp.vocab[csw].is_stop = True
    text_df = df[['sentence1', 'sentence2']]
    nlp_df = text_df.applymap(lambda x: nlp(x))
    tok_df = nlp_df.applymap(lambda doc: [token.text for token in doc if not token.is_stop])
    tok_df.columns = ['sentence1_tokens', 'sentence2_tokens']
    tokenized = pd.concat([df, tok_df], axis=1)
    return tokenized


