# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.


# This script reuses some code from
# https://github.com/huggingface/transformers

import logging
import os

import torch
from torch import nn

from transformers import BertModel, BertPreTrainedModel, PreTrainedModel, PretrainedConfig

from utils_nlp.dataset.url_utils import maybe_download
from utils_nlp.models.mtdnn.configuration_mtdnn import MTDNNConfig
from utils_nlp.models.mtdnn.utils.archive_maps import PRETRAINED_MODEL_ARCHIVE_MAP


logger = logging.getLogger(__name__)


class MTDNNPretrainedModel(BertPreTrainedModel):
    config_class = MTDNNConfig
    pretrained_model_archive_map = PRETRAINED_MODEL_ARCHIVE_MAP
    load_tf_weights = lambda model, config, path: None
    base_model_prefix = "mtdnn"

    def __init__(self, config):
        super(MTDNNPretrainedModel, self).__init__(config)
        if not isinstance(config, PretrainedConfig):
            raise ValueError(
                "Parameter config in `{}(config)` should be an instance of class `PretrainedConfig`. "
                "To create a model from a pretrained model use "
                "`model = {}.from_pretrained(PRETRAINED_MODEL_NAME)`".format(
                    self.__class__.__name__, self.__class__.__name__
                )
            )
        # Save config in model
        self.config = config


class MTDNNModel(MTDNNPretrainedModel, BertModel):
    def __init__(self, config: dict):
        super(MTDNNModel, self).__init__(config)

        self.task_types = config.pop("task_types", [])
        self.scoring_list = nn.ModuleList()
        self.labels = [int(ls) for ls in config["label_size"].split(",")]
        self.task_dropout_p = config["tasks_dropout_p"]

        for task, lab in enumerate(self.labels):
            task_type = self.task_types[task]
            dropout = DropoutWrapper(self.task_dropout_p[task], config["enable_variational_dropout"])
            self.dropout_list.append(dropout)
            if task_type == TaskType.Span:
                out_proj = nn.Linear(hidden_size, 2)
            elif task_type == TaskType.SeqenceLabeling:
                out_proj = nn.Linear(hidden_size, lab)
            else:
                out_proj = nn.Linear(hidden_size, lab)
            self.scoring_list.append(out_proj)

        # self.config = config
        self._my_init()

    def _my_init(self):
        def init_weights(module):
            if isinstance(module, (nn.Linear, nn.Embedding)):
                # Slightly different from the TF version which uses truncated_normal for initialization
                # cf https://github.com/pytorch/pytorch/pull/5617
                module.weight.data.normal_(mean=0.0, std=0.02 * self.opt["init_ratio"])
            elif isinstance(module, BertLayerNorm):
                # Slightly different from the BERT pytorch version, which should be a bug.
                # Note that it only affects on training from scratch. For detailed discussions, please contact xiaodl@.
                # Layer normalization (https://arxiv.org/abs/1607.06450)
                # support both old/latest version
                if "beta" in dir(module) and "gamma" in dir(module):
                    module.beta.data.zero_()
                    module.gamma.data.fill_(1.0)
                else:
                    module.bias.data.zero_()
                    module.weight.data.fill_(1.0)
            if isinstance(module, nn.Linear):
                module.bias.data.zero_()

        self.apply(init_weights)

    def forward(
        self, input_ids, token_type_ids, attention_mask, premise_mask=None, hyp_mask=None, task_id=0
    ):
        if self.encoder_type == EncoderModelType.ROBERTA:
            sequence_output = self.bert.extract_features(input_ids)
            pooled_output = self.pooler(sequence_output)
        else:
            all_encoder_layers, pooled_output = self.bert(input_ids, token_type_ids, attention_mask)
            sequence_output = all_encoder_layers[-1]

        decoder_opt = self.decoder_opt[task_id]
        task_type = self.task_types[task_id]
        if task_type == TaskType.Span:
            assert decoder_opt != 1
            sequence_output = self.dropout_list[task_id](sequence_output)
            logits = self.scoring_list[task_id](sequence_output)
            start_scores, end_scores = logits.split(1, dim=-1)
            start_scores = start_scores.squeeze(-1)
            end_scores = end_scores.squeeze(-1)
            return start_scores, end_scores
        elif task_type == TaskType.SeqenceLabeling:
            pooled_output = all_encoder_layers[-1]
            pooled_output = self.dropout_list[task_id](pooled_output)
            pooled_output = pooled_output.contiguous().view(-1, pooled_output.size(2))
            logits = self.scoring_list[task_id](pooled_output)
            return logits
        else:
            if decoder_opt == 1:
                max_query = hyp_mask.size(1)
                assert max_query > 0
                assert premise_mask is not None
                assert hyp_mask is not None
                hyp_mem = sequence_output[:, :max_query, :]
                logits = self.scoring_list[task_id](
                    sequence_output, hyp_mem, premise_mask, hyp_mask
                )
            else:
                pooled_output = self.dropout_list[task_id](pooled_output)
                logits = self.scoring_list[task_id](pooled_output)
            return logits


if __name__ == "__main__":
    config = MTDNNConfig()
    b = MTDNNModel(config)
    print(b.config_class)
    print(b.config)
    print(b.embeddings)
    print(b.encoder)
    print(b.pooler)
    print(b.pretrained_model_archive_map)
