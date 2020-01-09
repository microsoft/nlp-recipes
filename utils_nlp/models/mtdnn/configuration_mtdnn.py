# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.


# This script reuses some code from
# https://github.com/huggingface/transformers

from transformers import BertConfig, PretrainedConfig

"""MTDNN model configuration"""


class MTDNNConfig(PretrainedConfig):
    r"""
            :class:`~MTDNNConfig` is the configuration class to store the configuration of a
            `MTDNNModel`.


            Arguments:
                vocab_size: Vocabulary size of `inputs_ids` in `MTDNNModel`.
                hidden_size: Size of the encoder layers and the pooler layer.
                num_hidden_layers: Number of hidden layers in the Transformer encoder.
                num_attention_heads: Number of attention heads for each attention layer in
                    the Transformer encoder.
                intermediate_size: The size of the "intermediate" (i.e., feed-forward)
                    layer in the Transformer encoder.
                hidden_act: The non-linear activation function (function or string) in the
                    encoder and pooler. If string, "gelu", "relu", "swish" and "gelu_new" are supported.
                hidden_dropout_prob: The dropout probabilitiy for all fully connected
                    layers in the embeddings, encoder, and pooler.
                attention_probs_dropout_prob: The dropout ratio for the attention
                    probabilities.
                max_position_embeddings: The maximum sequence length that this model might
                    ever be used with. Typically set this to something large just in case
                    (e.g., 512 or 1024 or 2048).
                type_vocab_size: The vocabulary size of the `token_type_ids` passed into
                    `MTDNNModel`.
                initializer_range: The sttdev of the truncated_normal_initializer for
                    initializing all weight matrices.
                layer_norm_eps: The epsilon used by LayerNorm.
        """

    def __init__(
        self,
        vocab_size=30522,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        dump_feature=False,
        update_bert_opt=0,
        decoder_opts=[],
        label_size="",
        task_types=[],
        tasks_dropout_p=[],
        enable_variational_dropout=True,
        init_ratio=1.0,
        **kwargs,
    ):
        super(MTDNNConfig, self).__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.dump_feature = dump_feature
        self.update_bert_opt = update_bert_opt
        self.decoder_opts = decoder_opts
        self.label_size = label_size
        self.task_types = task_types
        self.tasks_dropout_p = tasks_dropout_p
        self.enable_variational_dropout = enable_variational_dropout
        self.init_ratio = init_ratio

