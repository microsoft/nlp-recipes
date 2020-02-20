# coding=utf-8
# Copyright (c) Microsoft. All rights reserved.
import random
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq.models.roberta import RobertaModel as FairseqRobertModel
from pytorch_pretrained_bert.modeling import BertConfig, BertLayerNorm, BertModel
from torch.nn.parameter import Parameter
from torch.nn.utils import weight_norm

from utils_nlp.models.mtdnn.common.dropout_wrapper import DropoutWrapper
from utils_nlp.models.mtdnn.common.optimizer import weight_norm as WN
from utils_nlp.models.mtdnn.common.similarity import FlatSimilarityWrapper, SelfAttnWrapper
from utils_nlp.models.mtdnn.common.types import EncoderModelType, TaskType
from utils_nlp.models.mtdnn.configuration_mtdnn import MTDNNConfig

SMALL_POS_NUM = 1.0e-30


class Classifier(nn.Module):
    def __init__(self, x_size, y_size, opt, prefix="decoder", dropout=None):
        super(Classifier, self).__init__()
        self.opt = opt
        if dropout is None:
            self.dropout = DropoutWrapper(opt.get("{}_dropout_p".format(prefix), 0))
        else:
            self.dropout = dropout
        self.merge_opt = opt.get("{}_merge_opt".format(prefix), 0)
        self.weight_norm_on = opt.get("{}_weight_norm_on".format(prefix), False)

        if self.merge_opt == 1:
            self.proj = nn.Linear(x_size * 4, y_size)
        else:
            self.proj = nn.Linear(x_size * 2, y_size)

        if self.weight_norm_on:
            self.proj = weight_norm(self.proj)

    def forward(self, x1, x2, mask=None):
        if self.merge_opt == 1:
            x = torch.cat([x1, x2, (x1 - x2).abs(), x1 * x2], 1)
        else:
            x = torch.cat([x1, x2], 1)
        x = self.dropout(x)
        scores = self.proj(x)
        return scores


class SANClassifier(nn.Module):
    """Implementation of Stochastic Answer Networks for Natural Language Inference, Xiaodong Liu, Kevin Duh and Jianfeng Gao
    https://arxiv.org/abs/1804.07888
    """

    def __init__(self, x_size, h_size, label_size, opt={}, prefix="decoder", dropout=None):
        super(SANClassifier, self).__init__()
        if dropout is None:
            self.dropout = DropoutWrapper(opt.get("{}_dropout_p".format(self.prefix), 0))
        else:
            self.dropout = dropout
        self.prefix = prefix
        self.query_wsum = SelfAttnWrapper(x_size, prefix="mem_cum", opt=opt, dropout=self.dropout)
        self.attn = FlatSimilarityWrapper(x_size, h_size, prefix, opt, self.dropout)
        self.rnn_type = "{}{}".format(opt.get("{}_rnn_type".format(prefix), "gru").upper(), "Cell")
        self.rnn = getattr(nn, self.rnn_type)(x_size, h_size)
        self.num_turn = opt.get("{}_num_turn".format(prefix), 5)
        self.opt = opt
        self.mem_random_drop = opt.get("{}_mem_drop_p".format(prefix), 0)
        self.mem_type = opt.get("{}_mem_type".format(prefix), 0)
        self.weight_norm_on = opt.get("{}_weight_norm_on".format(prefix), False)
        self.label_size = label_size
        self.dump_state = opt.get("dump_state_on", False)
        self.alpha = Parameter(torch.zeros(1, 1), requires_grad=False)
        if self.weight_norm_on:
            self.rnn = WN(self.rnn)

        self.classifier = Classifier(
            x_size, self.label_size, opt, prefix=prefix, dropout=self.dropout
        )

    def _generate_mask(self, new_data, dropout_p=0.0, is_training=False):
        if not is_training:
            dropout_p = 0.0
        new_data = (1 - dropout_p) * (new_data.zero_() + 1)
        for i in range(new_data.size(0)):
            one = random.randint(0, new_data.size(1) - 1)
            new_data[i][one] = 1
        mask = 1.0 / (1 - dropout_p) * torch.bernoulli(new_data)
        mask.requires_grad = False
        return mask

    def forward(self, x, h0, x_mask=None, h_mask=None):
        h0 = self.query_wsum(h0, h_mask)
        if type(self.rnn) is nn.LSTMCell:
            c0 = h0.new(h0.size()).zero_()
        scores_list = []
        for turn in range(self.num_turn):
            att_scores = self.attn(x, h0, x_mask)
            x_sum = torch.bmm(F.softmax(att_scores, 1).unsqueeze(1), x).squeeze(1)
            scores = self.classifier(x_sum, h0)
            scores_list.append(scores)
            # next turn
            if self.rnn is not None:
                h0 = self.dropout(h0)
                if type(self.rnn) is nn.LSTMCell:
                    h0, c0 = self.rnn(x_sum, (h0, c0))
                else:
                    h0 = self.rnn(x_sum, h0)
        if self.mem_type == 1:
            mask = self._generate_mask(
                self.alpha.data.new(x.size(0), self.num_turn), self.mem_random_drop, self.training
            )
            mask = [m.contiguous() for m in torch.unbind(mask, 1)]
            tmp_scores_list = [
                mask[idx].view(x.size(0), 1).expand_as(inp) * F.softmax(inp, 1)
                for idx, inp in enumerate(scores_list)
            ]
            scores = torch.stack(tmp_scores_list, 2)
            scores = torch.mean(scores, 2)
            scores = torch.log(scores)
        else:
            scores = scores_list[-1]
        if self.dump_state:
            return scores, scores_list
        else:
            return scores


class SANBERTNetwork(nn.Module):
    """Implementation of Stochastic Answer Networks for Natural Language Inference, Xiaodong Liu, Kevin Duh and Jianfeng Gao
    https://arxiv.org/abs/1804.07888
    """

    def __init__(
        self,
        init_checkpoint_model: Union[BertModel, FairseqRobertModel],
        pooler,
        config: MTDNNConfig,
    ):
        super(SANBERTNetwork, self).__init__()
        self.config = config
        self.bert = init_checkpoint_model
        self.pooler = pooler
        self.dropout_list = nn.ModuleList()
        self.encoder_type = config.encoder_type
        self.hidden_size = self.config.hidden_size

        # Dump other features if value is set to true
        if config.dump_feature:
            return

        # Update bert parameters
        if config.update_bert_opt > 0:
            for param in self.bert.parameters():
                param.requires_grad = False

        # Set decoder and scoring list parameters
        self.decoder_opts = config.decoder_opts
        self.scoring_list = nn.ModuleList()

        # Set task specific paramaters
        self.task_types = config.task_types
        self.task_dropout_p = config.tasks_dropout_p
        self.tasks_nclass_list = config.tasks_nclass_list

        # TODO - Move to training
        # Generate tasks decoding and scoring lists
        self._generate_tasks_decoding_scoring_options()

        # Initialize weights

        # self._my_init()

    def _my_init(self):
        def init_weights(module):
            if isinstance(module, (nn.Linear, nn.Embedding)):
                # Slightly different from the TF version which uses truncated_normal for initialization
                # cf https://github.com/pytorch/pytorch/pull/5617
                module.weight.data.normal_(mean=0.0, std=0.02 * self.config.init_ratio)
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
            all_encoder_layers, pooled_output = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
            sequence_output = all_encoder_layers[-1]

        decoder_opt = self.decoder_opts[task_id]
        task_type = self.task_types[task_id]
        if task_type == TaskType.Span:
            assert decoder_opt != 1
            sequence_output = self.dropout_list[task_id](sequence_output)
            logits = self.scoring_list[task_id](sequence_output)
            start_scores, end_scores = logits.split(1, dim=-1)
            start_scores = start_scores.squeeze(-1)
            end_scores = end_scores.squeeze(-1)
            return start_scores, end_scores
        elif task_type == TaskType.SequenceLabeling:
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

    # TODO - Move to training step
    def _generate_tasks_decoding_scoring_options(self):
        """ Enumerate over tasks and setup decoding and scoring list for training """
        assert len(self.tasks_nclass_list) > 0, "Number of classes to train for cannot be 0"
        for idx, task_num_labels in enumerate(self.tasks_nclass_list):
            print(f"idx: {idx}, number of task labels: {task_num_labels}")
            decoder_opt = self.decoder_opts[idx]
            task_type = self.task_types[idx]
            dropout = DropoutWrapper(
                self.task_dropout_p[idx], self.config.enable_variational_dropout
            )
            self.dropout_list.append(dropout)
            if task_type == TaskType.Span:
                assert decoder_opt != 1
                out_proj = nn.Linear(self.hidden_size, 2)
            elif task_type == TaskType.SequenceLabeling:
                out_proj = nn.Linear(self.hidden_size, task_num_labels)
            else:
                if decoder_opt == 1:
                    out_proj = SANClassifier(
                        self.hidden_size,
                        self.hidden_size,
                        task_num_labels,
                        self.config.to_dict(),
                        prefix="answer",
                        dropout=dropout,
                    )
                else:
                    out_proj = nn.Linear(self.hidden_size, task_num_labels)
            self.scoring_list.append(out_proj)
