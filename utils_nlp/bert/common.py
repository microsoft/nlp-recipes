# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from pytorch_pretrained_bert.tokenization import BertTokenizer
from enum import Enum
from tqdm import tqdm

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

        if isinstance(text[0], str):
            return [self.tokenizer.tokenize(x) for x in tqdm(text)]
        else:
            return [[self.tokenizer.tokenize(x) for x in sentences] for sentences in tqdm(text)]

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
        def truncate_and_add_sentence_marker(t):
            return

        if isinstance(tokens[0], str):
            tokens = [x[0 : max_len - 2] + ["[SEP]"] for x in tokens]
            token_type_ids = None
        else:

            def truncate_seq_pair(tokens_a, tokens_b, max_length):
                """Truncates a sequence pair in place to the maximum length."""
                # This is a simple heuristic which will always truncate the longer sequence
                # one token at a time. This makes more sense than truncating an equal percent
                # of tokens from each, since if one sequence is very short then each token
                # that's truncated likely contains more information than a longer sequence.
                while True:
                    total_length = len(tokens_a) + len(tokens_b)
                    if total_length <= max_length:
                        break
                    if len(tokens_a) > len(tokens_b):
                        tokens_a.pop()
                    else:
                        tokens_b.pop()

                tokens_a.append("[SEP]")
                tokens_b.append("[SEP]")

                return [tokens_a, tokens_b]

            # print(tokens[:2])
            # get tokens for each sentence [[t00, t01, ...] [t10, t11,... ]]
            tokens = [truncate_seq_pair(sentence[0], sentence[1], max_len - 3)  # [CLS] + 2x [SEP]
                      for sentence in tokens]

            # construct token_type_ids [[0, 0, 0, 0, ... 0, 1, 1, 1, ... 1], [0, 0, 0, ..., 1, 1, ]
            token_type_ids = [
                [[i] * len(sentence) for i, sentence in enumerate(example)]
                for example in tokens
            ]
            # merge sentences
            tokens = [[token for sentence in example for token in sentence]
                      for example in tokens]
            # prefix with [0] for [CLS]
            token_type_ids = [[0] + [i for sentence in example for i in sentence]
                              for example in token_type_ids]
            # pad sequence
            token_type_ids = [x + [0] * (max_len - len(x))
                              for x in token_type_ids]

        tokens = [["[CLS]"] + x for x in tokens]
        # convert tokens to indices
        tokens = [self.tokenizer.convert_tokens_to_ids(x) for x in tokens]
        # pad sequence
        tokens = [x + [0] * (max_len - len(x)) for x in tokens]
        # create input mask
        input_mask = [[min(1, x) for x in y] for y in tokens]
        return tokens, input_mask, token_type_ids
