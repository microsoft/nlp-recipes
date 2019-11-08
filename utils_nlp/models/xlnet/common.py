# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.


# This script reuses some code from
# https://github.com/huggingface/transformers/blob/master/examples/utils_glue.py
from enum import Enum
from transformers import XLNetTokenizer
from mlflow import log_metric, log_param, log_artifact


class Language(Enum):
    """
    An enumeration of the supported pretrained models and languages.
    """

    ENGLISHCASED = "xlnet-base-cased" #: Base cased model for xlnet
    ENGLISHLARGECASED = "xlnet-large-cased" #: Large cased model for xlnet

class Tokenizer:
    def __init__(
        self, language=Language.ENGLISHCASED, cache_dir="."
    ):
        """Initializes the underlying pretrained XLNet tokenizer.

        Args:
            language (Language, optional): The pretrained model's language.
                                           Defaults to Language.ENGLISHCASED
        """
        self.tokenizer = XLNetTokenizer.from_pretrained(language.value, cache_dir=cache_dir)
        self.language = language

    def preprocess_classification_tokens(self, examples, max_seq_length):
        """Preprocessing of example input tokens:
            - add XLNet sentence markers ([CLS] and [SEP])
            - pad and truncate sequences
            - create an input_mask
            - create token type ids, aka. segment ids

        Args:
            examples (list): List of input strings to preprocess.
            max_seq_length (int, optional): Maximum number of tokens
                            (documents will be truncated or padded).
                            Defaults to 512.
        Returns:
            (tuple): A tuple containing:
                list of input ids
                list of input mask
                list of segment ids

        """
        features = []
        cls_token = self.tokenizer.cls_token
        sep_token = self.tokenizer.sep_token
        cls_token_segment_id=2
        pad_on_left=True
        pad_token_segment_id=4
        sequence_a_segment_id=0
        cls_token_at_end=True
        mask_padding_with_zero=True
        pad_token=0
        
        list_input_ids = []
        list_input_mask = []
        list_segment_ids = []
        
        
        for (ex_index, example) in enumerate(examples):

            tokens_a = self.tokenizer.tokenize(example)

            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

            tokens = tokens_a + [sep_token]
            segment_ids = [sequence_a_segment_id] * len(tokens)

            if cls_token_at_end:
                tokens = tokens + [cls_token]
                segment_ids = segment_ids + [cls_token_segment_id]
            else:
                tokens = [cls_token] + tokens
                segment_ids = [cls_token_segment_id] + segment_ids

            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)


            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding_length = max_seq_length - len(input_ids)
            if pad_on_left:
                input_ids = ([pad_token] * padding_length) + input_ids
                input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
                segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
            else:
                input_ids = input_ids + ([pad_token] * padding_length)
                input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
                segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
          
            list_input_ids.append(input_ids)
            list_input_mask.append(input_mask)
            list_segment_ids.append(segment_ids)

#             features.append({"input_ids":input_ids,"input_mask":input_mask,"segment_ids":segment_ids,"label_id":label_id})
        return (list_input_ids, list_input_mask, list_segment_ids)

def log_xlnet_params(local_dict):
    """wrapper that abstracts away logging of ipython notebook local training parameters described at definition
    Args:
        local_dict(dict): dict containing all local varaibles from notebook 
    """
    params = ["DATA_FOLDER","XLNET_CACHE_DIR","LANGUAGE","MAX_SEQ_LENGTH","BATCH_SIZE","NUM_GPUS",
              "NUM_EPOCHS","TRAIN_SIZE","LABEL_COL","TEXT_COL","LEARNING_RATE","WEIGHT_DECAY",
              "ADAM_EPSILON","WARMUP_STEPS","DEBUG"]
    for i in params:
         log_param(i,local_dict[i])
    return