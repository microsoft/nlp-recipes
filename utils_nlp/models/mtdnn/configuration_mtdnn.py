# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.


# This script reuses some code from
# https://github.com/huggingface/transformers

import torch
from transformers import BertConfig, PretrainedConfig

from utils_nlp.models.mtdnn.common.types import EncoderModelType
from utils_nlp.models.mtdnn.common.archive_maps import PRETRAINED_CONFIG_ARCHIVE_MAP

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

    # TODO - Not needed
    pretrained_config_archive_map = PRETRAINED_CONFIG_ARCHIVE_MAP

    def __init__(
        self,
        encoder_type=EncoderModelType.BERT,
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
        n_class=2,
        task_types=[],
        tasks_dropout_p=[],
        enable_variational_dropout=True,
        init_ratio=1.0,
        init_checkpoint="bert-base-uncased",
        # Training config
        cuda=torch.cuda.is_available(),
        multi_gpu_on=False,
        log_per_updates=500,
        save_per_updates=10000,
        save_per_updates_on=False,
        epochs=5,
        batch_size=8,
        batch_size_eval=8,
        optimizer="adamax",
        grad_clipping=0.0,
        global_grad_clipping=1.0,
        weight_decay=0.0,
        learning_rate=5e-5,
        momentum=0.0,
        warmup=0.1,
        warmup_schedule="warmup_linear",
        adam_eps=1e-6,
        # Scheduler config
        have_lr_scheduler=True,
        multi_step_lr="10,20,30",
        freeze_layers=1,
        embedding_opt=0,
        lr_gamma=0.5,
        bert_l2norm=0.0,
        scheduler_type="ms",
        seed=2018,
        grad_accumulation_step=1,
        # fp16
        fp16=False,
        fp16_opt_level="01" ** kwargs,
    ):
        super(MTDNNConfig, self).__init__(**kwargs)
        self.encoder_type = encoder_type
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
        self.n_class = n_class
        self.task_types = task_types
        self.tasks_dropout_p = tasks_dropout_p
        self.enable_variational_dropout = enable_variational_dropout
        self.init_ratio = init_ratio
        self.init_checkpoint = init_checkpoint
        self.cuda = cuda
        self.multi_gpu_on = multi_gpu_on
        self.log_per_updates = log_per_updates
        self.save_per_updates = save_per_updates
        self.save_per_updates_on = save_per_updates_on
        self.epochs = epochs
        self.batch_size = batch_size
        self.batch_size_eval = batch_size_eval
        self.optimizer = optimizer
        self.grad_clipping = grad_clipping
        self.global_grad_clipping = global_grad_clipping
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.warmup = warmup
        self.warmup_schedule = warmup_schedule
        self.adam_eps = adam_eps
        self.have_lr_scheduler = have_lr_scheduler
        self.multi_step_lr = multi_step_lr
        self.freeze_layers = freeze_layers
        self.embedding_opt = embedding_opt
        self.lr_gamma = lr_gamma
        self.bert_l2norm = bert_l2norm
        self.scheduler_type = scheduler_type
        self.seed = seed
        self.grad_accumulation_step = grad_accumulation_step
        self.fp16 = fp16
        self.kwargs = kwargs

