# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# This script reuses some code from
# https://github.com/huggingface/pytorch-transformers/blob/master/examples/run_glue.py

import logging
import os
import random

import numpy as np
import torch
from tqdm import tqdm, trange
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

from transformers.modeling_bert import BERT_PRETRAINED_MODEL_ARCHIVE_MAP
from transformers.modeling_distilbert import DISTILBERT_PRETRAINED_MODEL_ARCHIVE_MAP
from transformers.modeling_roberta import ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
from transformers.modeling_xlnet import XLNET_PRETRAINED_MODEL_ARCHIVE_MAP
from transformers.tokenization_bert import BertTokenizer
from transformers.tokenization_distilbert import DistilBertTokenizer
from transformers.tokenization_roberta import RobertaTokenizer
from transformers.tokenization_xlnet import XLNetTokenizer
from utils_nlp.common.pytorch_utils import get_device

TOKENIZER_CLASS = {}
TOKENIZER_CLASS.update({k: BertTokenizer for k in BERT_PRETRAINED_MODEL_ARCHIVE_MAP})
TOKENIZER_CLASS.update({k: RobertaTokenizer for k in ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP})
TOKENIZER_CLASS.update({k: XLNetTokenizer for k in XLNET_PRETRAINED_MODEL_ARCHIVE_MAP})
TOKENIZER_CLASS.update({k: DistilBertTokenizer for k in DISTILBERT_PRETRAINED_MODEL_ARCHIVE_MAP})

MAX_SEQ_LEN = 512

logger = logging.getLogger(__name__)


class Transformer:
    def __init__(
        self,
        model_class,
        model_name="bert-base-cased",
        num_labels=2,
        cache_dir=".",
        load_model_from_dir=None,
    ):

        if model_name not in self.list_supported_models():
            raise ValueError(
                "Model name {0} is not supported by {1}. "
                "Call '{1}.list_supported_models()' to get all supported model "
                "names.".format(value, self.__class__.__name__)
            )
        self._model_name = model_name
        self._model_type = model_name.split("-")[0]
        self.cache_dir = cache_dir
        self.load_model_from_dir = load_model_from_dir
        if load_model_from_dir is None:
            self.model = model_class[model_name].from_pretrained(
                model_name, cache_dir=cache_dir, num_labels=num_labels, output_loading_info=False
            )
        else:
            logger.info("Loading cached model from {}".format(load_model_from_dir))
            self.model = model_class[model_name].from_pretrained(
                load_model_from_dir, num_labels=num_labels, output_loading_info=False
            )

    @property
    def model_name(self):
        return self._model_name

    @property
    def model_type(self):
        return self._model_type

    @staticmethod
    def set_seed(seed, cuda=True):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if cuda and torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def fine_tune(
        self,
        train_dataloader,
        get_inputs,
        max_steps=-1,
        num_train_epochs=1,
        max_grad_norm=1.0,
        gradient_accumulation_steps=1,
        n_gpu=1,
        optimizer=None,
        scheduler=None,
        weight_decay=0.0,
        learning_rate=5e-5,
        adam_epsilon=1e-8,
        warmup_steps=0,
        fp16=False,
        fp16_opt_level="O1",
        local_rank=-1,
        verbose=True,
        seed=None,
    ):

        device, num_gpus = get_device(num_gpus=n_gpu, local_rank=-1)

        if seed is not None:
            Transformer.set_seed(seed, num_gpus > 0)

        if max_steps > 0:
            t_total = max_steps
            num_train_epochs = (
                max_steps // (len(train_dataloader) // gradient_accumulation_steps) + 1
            )
        else:
            t_total = len(train_dataloader) // gradient_accumulation_steps * num_train_epochs

        if optimizer is None:
            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p
                        for n, p in self.model.named_parameters()
                        if not any(nd in n for nd in no_decay)
                    ],
                    "weight_decay": weight_decay,
                },
                {
                    "params": [
                        p
                        for n, p in self.model.named_parameters()
                        if any(nd in n for nd in no_decay)
                    ],
                    "weight_decay": 0.0,
                },
            ]
            optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)

        if scheduler is None:
            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
            )

        if fp16:
            try:
                from apex import amp
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex")
            self.model, optimizer = amp.initialize(self.model, optimizer, opt_level=fp16_opt_level)

        if local_rank != -1:
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=True,
            )
        else:
            if isinstance(self.model, torch.nn.DataParallel):
                self.model = self.model.module

            if num_gpus > 1:
                self.model = torch.nn.DataParallel(self.model, device_ids=list(range(num_gpus)))

        self.model.to(device)
        self.model.train()

        global_step = 0
        tr_loss = 0.0
        self.model.zero_grad()
        train_iterator = trange(
            int(num_train_epochs), desc="Epoch", disable=local_rank not in [-1, 0] or not verbose
        )

        for _ in train_iterator:
            epoch_iterator = tqdm(
                train_dataloader, desc="Iteration", disable=local_rank not in [-1, 0] or not verbose
            )
            for step, batch in enumerate(epoch_iterator):
                batch = tuple(t.to(device) for t in batch)
                inputs = get_inputs(batch, self.model_name)
                outputs = self.model(**inputs)
                loss = outputs[0]

                if num_gpus > 1:
                    loss = loss.mean()
                if gradient_accumulation_steps > 1:
                    loss = loss / gradient_accumulation_steps

                if step % 10 == 0 and verbose:
                    tqdm.write("Loss:{:.6f}".format(loss))

                if fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), max_grad_norm)
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)

                tr_loss += loss.item()
                if (step + 1) % gradient_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()
                    self.model.zero_grad()
                    global_step += 1

                if max_steps > 0 and global_step > max_steps:
                    epoch_iterator.close()
                    break
            if max_steps > 0 and global_step > max_steps:
                train_iterator.close()
                break

            # empty cache
            del [batch]
            torch.cuda.empty_cache()
        return global_step, tr_loss / global_step

    def predict(self, eval_dataloader, get_inputs, n_gpu=1, verbose=True):
        device, num_gpus = get_device(num_gpus=n_gpu, local_rank=-1)

        if isinstance(self.model, torch.nn.DataParallel):
            self.model = self.model.module

        if num_gpus > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=list(range(num_gpus)))

        self.model.to(device)
        self.model.eval()

        for batch in tqdm(eval_dataloader, desc="Evaluating", disable=not verbose):
            batch = tuple(t.to(device) for t in batch)
            with torch.no_grad():
                inputs = get_inputs(batch, self.model_name, train_mode=False)
                outputs = self.model(**inputs)
                logits = outputs[0]
            yield logits.detach().cpu().numpy()

    def save_model(self):
        output_model_dir = os.path.join(self.cache_dir, "fine_tuned")

        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(output_model_dir, exist_ok=True)

        logger.info("Saving model checkpoint to %s", output_model_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = (
            self.model.module if hasattr(self.model, "module") else self.model
        )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(output_model_dir)
