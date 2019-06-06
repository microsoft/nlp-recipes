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
        """Initializes the underlying pretrained BERT tokenizer.
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
        """Uses a BERT tokenizer 
        
        Args:
            text (list): [description]
        
        Returns:
            [list]: [description]
        """
        tokens = [self.tokenizer.tokenize(x) for x in text]
        return tokens

    def preprocess_classification_tokens(self, tokens, max_len=BERT_MAX_LEN):
        """Preprocessing of input tokens:
            - add BERT sentence markers ([CLS] and [SEP])
            - map tokens to indices
            - pad and truncate sequences
            - create an input_mask    
        Args:
            tokens (list): List of tokens to preprocess.
            max_len (int, optional): Maximum number of tokens
                            (documents will be truncated or padded).
                            Defaults to 512.
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
        tokens = [["[CLS]"] + x[0 : max_len - 2] + ["[SEP]"] for x in tokens]
        # convert tokens to indices
        tokens = [self.tokenizer.convert_tokens_to_ids(x) for x in tokens]
        # pad sequence
        tokens = [x + [0] * (max_len - len(x)) for x in tokens]
        # create input mask
        input_mask = [[min(1, x) for x in y] for y in tokens]
        return tokens, input_mask
