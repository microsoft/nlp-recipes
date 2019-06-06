# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from pytorch_pretrained_bert.tokenization import BertTokenizer
from enum import Enum

# Max supported sequence length
BERT_MAX_LEN = 512


class Language(Enum):
    """An enumeration of the supported languages."""

    ENGLISH = "bert-base-uncased"
    ENGLISHCASED = "bert-base-cased"
    ENGLISHLARGE = "bert-large-uncased"
    ENGLISHLARGECASED = "bert-large-cased"
    CHINESE = "bert-base-chinese"
    MULTILINGUAL = "bert-base-multilingual-cased"


class Tokenizer:
    def __init__(
        self, language=Language.ENGLISH, to_lower=False, cache_dir="."
    ):
        """Initializes the tokenizer and the underlying pretrained tokenizer.
        Args:
            language (Language, optional): The pretrained model's language.
                                           Defaults to Language.ENGLISH.
            cache_dir (str, optional): Location of BERT's cache directory. Defaults to ".".
        """
        self.tokenizer = BertTokenizer.from_pretrained(
            language.value, do_lower_case=to_lower, cache_dir=cache_dir
        )
        self.language = language

    def tokenize(self, text):
        # TODO: reload module
        # TODO: check for text not to be string... 
        try:
            tokens = [[self.tokenizer.tokenize(x) for x in sentences] for sentences in text]
        except TypeError:
            tokens = [self.tokenizer.tokenize(x) for x in text]
        return tokens

    def preprocess_classification_tokens(self, tokens, max_len):
        """Preprocessing of input tokens:
            - add BERT sentence markers ([CLS] and [SEP])
            - map tokens to indices
            - pad and truncate sequences 
            - create an input_mask    
        Args:
            tokens ([type]): List of tokens to preprocess.
            max_len ([type]): Maximum length of sequence.        
        Returns:
            list of preprocesssed token lists
            list of input mask lists
        """
        if max_len > BERT_MAX_LEN:
            print(
                "setting max_len to max allowed tokens: {}".format(
                    BERT_MAX_LEN
                )
            )
            max_len = BERT_MAX_LEN

        # truncate and add BERT sentence markers
        def truncate_and_add_sentence_marker(t):
            return [x[0 : max_len - 2] + ["[SEP]"] for x in t]

        # duck-typing... https://stackoverflow.com/questions/1952464/in-python-how-do-i-determine-if-an-object-is-iterable
        try:
            # get tokens for each sentence [[t00, t01, ...] [t10, t11,... ]] 
            tokens = [truncate_and_add_sentence_marker(sentence) for sentence in tokens]
            # construct token_type_ids [0, 0, 0, 0, ... 0, 1, 1, 1, ... 1]
            token_type_ids = [id for id, sentence in enumerate(tokens) for _ in range(len(sentence))]
            # flatten the tokens
            tokens = [t for sentence in tokens for t in sentence]
        except TypeError:
            tokens = truncate_and_add_sentence_marker(tokens)
            token_type_ids = None

        tokens = ["[CLS]"] + tokens
        # convert tokens to indices
        tokens = [self.tokenizer.convert_tokens_to_ids(x) for x in tokens]
        # pad sequence
        tokens = [x + [0] * (max_len - len(x)) for x in tokens]
        # create input mask
        input_mask = [[min(1, x) for x in y] for y in tokens]
        return tokens, input_mask, token_type_ids
