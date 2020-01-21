# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.


# This script reuses some code from
# https://github.com/huggingface/transformers

import logging
import os
import pathlib
import sys

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from fairseq.models.roberta import RobertaModel as FairseqRobertModel
from torch import nn
from torch.optim.lr_scheduler import *
from transformers import (
    BertConfig,
    BertModel,
    BertPreTrainedModel,
    PretrainedConfig,
    PreTrainedModel,
    RobertaModel,
)

from utils_nlp.dataset.url_utils import download_path, maybe_download
from utils_nlp.models.mtdnn.common.archive_maps import PRETRAINED_MODEL_ARCHIVE_MAP
from utils_nlp.models.mtdnn.common.average_meter import AverageMeter
from utils_nlp.models.mtdnn.common.bert_optim import Adamax, RAdam
from utils_nlp.models.mtdnn.common.linear_pooler import LinearPooler
from utils_nlp.models.mtdnn.common.loss import LOSS_REGISTRY
from utils_nlp.models.mtdnn.common.san import SANClassifier, SANNetwork
from utils_nlp.models.mtdnn.common.squad_utils import extract_answer
from utils_nlp.models.mtdnn.common.types import DataFormat, EncoderModelType, TaskType
from utils_nlp.models.mtdnn.common.utils import MTDNNCommonUtils
from utils_nlp.models.mtdnn.configuration_mtdnn import MTDNNConfig

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
    def __init__(
        self,
        config: MTDNNConfig,
        pretrained_model_name: str = "mtdnn-base-uncased",
        num_train_step: int = -1,
    ):
        super(MTDNNModel, self).__init__(config)
        self.config = config

        # Set the config base on encoder type set for initial checkpoint

        # Download pretrained model
        # TODO - Use Model.pretrained_model() after configuration file is hosted.
        with download_path() as file_path:
            path = pathlib.Path(file_path)
            self.local_model_path = maybe_download(
                url=self.pretrained_model_archive_map[pretrained_model_name]
            )
        self.mtdnn_model = MTDNNCommonUtils.load_pytorch_model(self.local_model_path)

        self.state_dict = self.mtdnn_model["state"]
        self.updates = (
            self.state_dict["updates"] if self.state_dict and "updates" in self.state_dict else 0
        )
        self.local_updates = 0
        self.train_loss = AverageMeter()
        self.network = SANNetwork(self.config)
        if self.state_dict:
            self.network.load_state_dict(self.state_dict, strict=False)
        self.mnetwork = nn.DataParallel(self.network) if self.config.multi_gpu_on else self.network
        self.total_param = sum([p.nelement() for p in self.network.parameters() if p.requires_grad])

        # Move network to GPU if device available and flag set
        print(f" =======> Can move to cuda {self.config.cuda} and {torch.cuda.is_available()}")
        if self.config.cuda:
            print(" =======> Moving to cuda")
            self.network.cuda()
        self.optimizer_parameters = self._get_param_groups()
        self._setup_optim(self.optimizer_parameters, self.state_dict, num_train_step)
        self.para_swapped = False
        self.optimizer.zero_grad()
        self._setup_lossmap()

    def _get_param_groups(self):
        no_decay = ["bias", "gamma", "beta", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_parameters = [
            {
                "params": [
                    p
                    for n, p in self.network.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.01,
            },
            {
                "params": [
                    p for n, p in self.network.named_parameters() if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        return optimizer_parameters

    def _setup_optim(self, optimizer_parameters, state_dict: dict = None, num_train_step: int = -1):

        # Setup optimizer parameters
        if self.config.optimizer == "sgd":
            self.optimizer = optim.SGD(
                optimizer_parameters,
                self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        elif self.config.optimizer == "adamax":
            self.optimizer = Adamax(
                optimizer_parameters,
                self.config.learning_rate,
                warmup=self.config.warmup,
                t_total=num_train_step,
                max_grad_norm=self.config.grad_clipping,
                schedule=self.config.warmup_schedule,
                weight_decay=self.config.weight_decay,
            )

        elif self.config.optimizer == "radam":
            self.optimizer = RAdam(
                optimizer_parameters,
                self.config.learning_rate,
                warmup=self.config.warmup,
                t_total=num_train_step,
                max_grad_norm=self.config.grad_clipping,
                schedule=self.config.warmup_schedule,
                eps=self.config.adam_eps,
                weight_decay=self.config.weight_decay,
            )

            # The current radam does not support FP16.
            self.config.fp16 = False
        elif self.config.optimizer == "adam":
            self.optimizer = Adam(
                optimizer_parameters,
                lr=self.config.learning_rate,
                warmup=self.config.warmup,
                t_total=num_train_step,
                max_grad_norm=self.config.grad_clipping,
                schedule=self.config.warmup_schedule,
                weight_decay=self.config.weight_decay,
            )

        else:
            raise RuntimeError(f"Unsupported optimizer: {self.config.optimizer}")

        # Clear scheduler for certain optimizer choices
        if self.config.optimizer in ["adam", "adamax", "radam"]:
            if self.config.have_lr_scheduler:
                self.config.have_lr_scheduler = False

        if state_dict and "optimizer" in state_dict:
            self.optimizer.load_state_dict(state_dict["optimizer"])

        if self.config.fp16:
            try:
                from apex import amp

                global amp
            except ImportError:
                raise ImportError(
                    "Please install apex from https://www.github.com/nvidia/apex to use fp16 training."
                )
            model, optimizer = amp.initialize(
                self.network, self.optimizer, opt_level=self.config.fp16_opt_level
            )
            self.network = model
            self.optimizer = optimizer

        if self.config.have_lr_scheduler:
            if self.config.scheduler_type == "rop":
                self.scheduler = ReduceLROnPlateau(
                    self.optimizer, mode="max", factor=self.config.lr_gamma, patience=3
                )
            elif self.config.scheduler_type == "exp":
                self.scheduler = ExponentialLR(self.optimizer, gamma=self.config.lr_gamma or 0.95)
            else:
                milestones = [
                    int(step) for step in (self.config.multi_step_lr or "10,20,30").split(",")
                ]
                self.scheduler = MultiStepLR(
                    self.optimizer, milestones=milestones, gamma=self.config.lr_gamma
                )
        else:
            self.scheduler = None

    def _setup_lossmap(self):
        loss_types = self.config.loss_types
        self.task_loss_criterion = []
        for idx, cs in enumerate(loss_types):
            assert cs, "Loss type must be defined."
            lc = LOSS_REGISTRY[cs](name="Loss func of task {}: {}".format(idx, cs))
            self.task_loss_criterion.append(lc)

    def _setup_kd_lossmap(self):
        loss_types = self.config.kd_loss_types
        self.kd_task_loss_criterion = []
        if config.mkd_opt > 0:
            for idx, cs in enumerate(loss_types):
                assert cs, "Loss type must be defined."
                lc = LOSS_REGISTRY[cs](name="Loss func of task {}: {}".format(idx, cs))
                self.kd_task_loss_criterion.append(lc)

    def train(self):
        if self.para_swapped:
            self.para_swapped = False

    def _to_cuda(self, tensor):
        if not tensor:
            return tensor

        if isinstance(tensor, list) or isinstance(tensor, tuple):
            y = [e.cuda(non_blocking=True) for e in tensor]
            for e in y:
                e.requires_grad = False
        else:
            y = tensor.cuda(non_blocking=True)
            y.requires_grad = False
        return y

    def update(self, batch_meta, batch_data):
        self.network.train()
        y = batch_data[batch_meta["label"]]
        soft_labels = None

        task_type = batch_meta["task_type"]
        y = self._to_cuda(y) if self.config.cuda else y

        task_id = batch_meta["task_id"]
        inputs = batch_data[: batch_meta["input_len"]]
        if len(inputs) == 3:
            inputs.append(None)
            inputs.append(None)
        inputs.append(task_id)
        weight = None
        if self.config.weighted_on:
            if self.config.cuda:
                weight = batch_data[batch_meta["factor"]].cuda(non_blocking=True)
            else:
                weight = batch_data[batch_meta["factor"]]
        logits = self.mnetwork(*inputs)

        # compute loss
        loss = 0
        if self.task_loss_criterion[task_id] and (y is not None):
            loss = self.task_loss_criterion[task_id](logits, y, weight, ignore_index=-1)

        # compute kd loss
        if self.config.get("mkd_opt", 0) > 0 and ("soft_label" in batch_meta):
            soft_labels = batch_meta["soft_label"]
            soft_labels = self._to_cuda(soft_labels) if self.config.cuda else soft_labels
            kd_lc = self.kd_task_loss_criterion[task_id]
            kd_loss = kd_lc(logits, soft_labels, weight, ignore_index=-1) if kd_lc else 0
            loss = loss + kd_loss

        self.train_loss.update(loss.item(), batch_data[batch_meta["token_id"]].size(0))
        # scale loss
        loss = loss / self.config.get("grad_accumulation_step", 1)
        if self.config.fp16:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        self.local_updates += 1
        if self.local_updates % self.config.grad_accumulation_step == 0:
            if self.config.global_grad_clipping > 0:
                if self.config.fp16:
                    torch.nn.utils.clip_grad_norm_(
                        amp.master_params(self.optimizer), self.config.global_grad_clipping
                    )
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.network.parameters(), self.config.global_grad_clipping
                    )
            self.updates += 1
            # reset number of the grad accumulation
            self.optimizer.step()
            self.optimizer.zero_grad()

    def predict(self, batch_meta, batch_data):
        self.network.eval()
        task_id = batch_meta["task_id"]
        task_type = batch_meta["task_type"]
        inputs = batch_data[: batch_meta["input_len"]]
        if len(inputs) == 3:
            inputs.append(None)
            inputs.append(None)
        inputs.append(task_id)
        score = self.mnetwork(*inputs)
        if task_type == TaskType.Ranking:
            score = score.contiguous().view(-1, batch_meta["pairwise_size"])
            assert task_type == TaskType.Ranking
            score = F.softmax(score, dim=1)
            score = score.data.cpu()
            score = score.numpy()
            predict = np.zeros(score.shape, dtype=int)
            positive = np.argmax(score, axis=1)
            for idx, pos in enumerate(positive):
                predict[idx, pos] = 1
            predict = predict.reshape(-1).tolist()
            score = score.reshape(-1).tolist()
            return score, predict, batch_meta["true_label"]
        elif task_type == TaskType.SequenceLabeling:
            mask = batch_data[batch_meta["mask"]]
            score = score.contiguous()
            score = score.data.cpu()
            score = score.numpy()
            predict = np.argmax(score, axis=1).reshape(mask.size()).tolist()
            valied_lenght = mask.sum(1).tolist()
            final_predict = []
            for idx, p in enumerate(predict):
                final_predict.append(p[: valied_lenght[idx]])
            score = score.reshape(-1).tolist()
            return score, final_predict, batch_meta["label"]
        elif task_type == TaskType.Span:
            start, end = score
            predictions = []
            if self.config.encoder_type == EncoderModelType.BERT:
                scores, predictions = extract_answer(
                    batch_meta, batch_data, start, end, self.config.get("max_answer_len", 5)
                )
            return scores, predictions, batch_meta["answer"]
        else:
            if task_type == TaskType.Classification:
                score = F.softmax(score, dim=1)
            score = score.data.cpu()
            score = score.numpy()
            predict = np.argmax(score, axis=1).tolist()
            score = score.reshape(-1).tolist()
        return score, predict, batch_meta["label"]

    def extract(self, batch_meta, batch_data):
        self.network.eval()
        # 'token_id': 0; 'segment_id': 1; 'mask': 2
        inputs = batch_data[:3]
        all_encoder_layers, pooled_output = self.mnetwork.bert(*inputs)
        return all_encoder_layers, pooled_output

    def save(self, filename):
        network_state = dict([(k, v.cpu()) for k, v in self.network.state_dict().items()])
        params = {
            "state": network_state,
            "optimizer": self.optimizer.state_dict(),
            "config": self.config,
        }
        torch.save(params, filename)
        logger.info("model saved to {}".format(filename))

    def load(self, checkpoint):
        model_state_dict = torch.load(checkpoint)
        self.network.load_state_dict(model_state_dict["state"], strict=False)
        self.optimizer.load_state_dict(model_state_dict["optimizer"])
        self.config.update(model_state_dict["config"])

    def cuda(self):
        self.network.cuda()
