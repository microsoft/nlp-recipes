# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# This script reuses code from:
# https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/examples
# /extract_features.py, with necessary modifications.

from enum import Enum

import numpy as np
import pandas as pd
import torch
from cached_property import cached_property
from pytorch_pretrained_bert.modeling import BertModel
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset

from utils_nlp.common.pytorch_utils import (
    get_device,
    move_model_to_device,
    parallelize_model,
)
from utils_nlp.models.bert.common import Language, Tokenizer


class PoolingStrategy(str, Enum):
    """Enumerate pooling strategies"""

    MAX: str = "max"
    MEAN: str = "mean"
    CLS: str = "cls"


class BERTSentenceEncoder:
    """BERT-based sentence encoder"""

    def __init__(
        self,
        bert_model=None,
        tokenizer=None,
        language=Language.ENGLISH,
        num_gpus=None,
        cache_dir=".",
        to_lower=True,
        max_len=512,
        layer_index=-1,
        pooling_strategy=PoolingStrategy.MEAN,
    ):
        """Initialize the encoder's underlying model and tokenizer

        Args:
            bert_model: BERT model to use for encoding.
                Defaults to pretrained BertModel.
            tokenizer: Tokenizer to use for preprocessing.
                Defaults to pretrained BERT tokenizer.
            language: The pretrained model's language. Defaults to Language.ENGLISH.
            num_gpus: The number of gpus to use. Defaults to None, which forces all
                available GPUs to be used.
            cache_dir: Location of BERT's cache directory. Defaults to "."
            to_lower: True to lowercase before tokenization. Defaults to False.
            max_len: Maximum number of tokens.
            layer_index: The layer from which to extract features.
                         Defaults to the last layer; can also be a list of integers
                         for experimentation.
            pooling_strategy: Pooling strategy to aggregate token embeddings into
                sentence embedding.
        """
        self.model = (
            bert_model.model.bert
            if bert_model
            else BertModel.from_pretrained(language, cache_dir=cache_dir)
        )
        self.tokenizer = (
            tokenizer
            if tokenizer
            else Tokenizer(language, to_lower=to_lower, cache_dir=cache_dir)
        )
        self.num_gpus = num_gpus
        self.max_len = max_len
        self.layer_index = layer_index
        self.pooling_strategy = pooling_strategy
        self.has_cuda = self.cuda

    @property
    def layer_index(self):
        return self._layer_index

    @layer_index.setter
    def layer_index(self, layer_index):
        if isinstance(layer_index, int):
            self._layer_index = [layer_index]
        else:
            self.layer_index = layer_index

    @cached_property
    def cuda(self):
        """ cache the output of torch.cuda.is_available() """

        self.has_cuda = torch.cuda.is_available()
        return self.has_cuda

    @property
    def pooling_strategy(self):
        return self._pooling_strategy

    @pooling_strategy.setter
    def pooling_strategy(self, pooling_strategy):
        self._pooling_strategy = pooling_strategy

    def get_hidden_states(self, text, batch_size=32):
        """Extract the hidden states from the pretrained model

        Args:
            text: List of documents to extract features from.
            batch_size: Batch size, defaults to 32.

        Returns:
            pd.DataFrame with columns:
                text_index (int), token (str), layer_index (int), values (list[float]).
        """
        device, num_gpus = get_device(self.num_gpus)
        self.model = move_model_to_device(self.model, device)
        self.model = parallelize_model(self.model, device, self.num_gpus)

        self.model.eval()

        tokens = self.tokenizer.tokenize(text)

        (
            tokens,
            input_ids,
            input_mask,
            input_type_ids,
        ) = self.tokenizer.preprocess_encoder_tokens(tokens, max_len=self.max_len)

        input_ids = torch.tensor(input_ids, dtype=torch.long, device=device)
        input_mask = torch.tensor(input_mask, dtype=torch.long, device=device)
        input_type_ids = torch.arange(
            input_ids.size(0), dtype=torch.long, device=device
        )

        eval_data = TensorDataset(input_ids, input_mask, input_type_ids)
        eval_dataloader = DataLoader(
            eval_data, sampler=SequentialSampler(eval_data), batch_size=batch_size
        )

        hidden_states = {"text_index": [], "token": [], "layer_index": [], "values": []}
        for (
            input_ids_tensor,
            input_mask_tensor,
            example_indices_tensor,
        ) in eval_dataloader:
            with torch.no_grad():
                all_encoder_layers, _ = self.model(
                    input_ids_tensor,
                    token_type_ids=None,
                    attention_mask=input_mask_tensor,
                )
                self.embedding_dim = all_encoder_layers[0].size()[-1]

            for b, example_index in enumerate(example_indices_tensor):
                for (i, token) in enumerate(tokens[example_index.item()]):
                    for (j, layer_index) in enumerate(self.layer_index):
                        layer_output = (
                            all_encoder_layers[int(layer_index)].detach().cpu().numpy()
                        )
                        layer_output = layer_output[b]
                        hidden_states["text_index"].append(example_index.item())
                        hidden_states["token"].append(token)
                        hidden_states["layer_index"].append(layer_index)
                        hidden_states["values"].append(
                            [round(x.item(), 6) for x in layer_output[i]]
                        )

            # empty cache
            del [input_ids_tensor, input_mask_tensor, example_indices_tensor]
            torch.cuda.empty_cache()

        # empty cache
        del [input_ids, input_mask, input_type_ids]
        torch.cuda.empty_cache()

        return pd.DataFrame.from_dict(hidden_states)

    def pool(self, df):
        """Pooling to aggregate token-wise embeddings to sentence embeddings

        Args:
            df: pd.DataFrame with columns text_index (int), token (str),
                layer_index (int), values (list[float])

        Returns:
            pd.DataFrame grouped by text index and layer index
        """

        def max_pool(x):
            values = np.array(
                [
                    np.reshape(np.array(x.values[i]), self.embedding_dim)
                    for i in range(x.values.shape[0])
                ]
            )
            m, _ = torch.max(torch.tensor(values, dtype=torch.float), 0)
            return m.numpy()

        def mean_pool(x):
            values = np.array(
                [
                    np.reshape(np.array(x.values[i]), self.embedding_dim)
                    for i in range(x.values.shape[0])
                ]
            )
            return torch.mean(torch.tensor(values, dtype=torch.float), 0).numpy()

        def cls_pool(x):
            values = np.array(
                [
                    np.reshape(np.array(x.values[i]), self.embedding_dim)
                    for i in range(x.values.shape[0])
                ]
            )
            return values[0]

        try:
            if self.pooling_strategy == "max":
                pool_func = max_pool
            elif self.pooling_strategy == "mean":
                pool_func = mean_pool
            elif self.pooling_strategy == "cls":
                pool_func = cls_pool
            else:
                raise ValueError("Please enter valid pooling strategy")
        except ValueError as ve:
            print(ve)

        return (
            df.groupby(["text_index", "layer_index"])["values"]
            .apply(lambda x: pool_func(x))
            .reset_index()
        )

    def encode(self, text, batch_size=32, as_numpy=False):
        """Computes sentence encodings

        Args:
            text: List of documents to encode.
            batch_size: Batch size, defaults to 32.
        """
        df = self.get_hidden_states(text, batch_size)
        pooled = self.pool(df)

        if as_numpy:
            return np.array(pooled["values"].tolist())
        else:
            return pooled
