# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# This script reuses some code from
# https://github.com/huggingface/pytorch-transformers/blob/master/examples/run_glue.py

import datetime
import logging
import os
import random
import time

import numpy as np
import torch
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers.modeling_bert import BERT_PRETRAINED_MODEL_ARCHIVE_MAP
from transformers.modeling_distilbert import DISTILBERT_PRETRAINED_MODEL_ARCHIVE_MAP
from transformers.modeling_roberta import ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
from transformers.modeling_xlnet import XLNET_PRETRAINED_MODEL_ARCHIVE_MAP
from transformers.tokenization_bert import BertTokenizer
from transformers.tokenization_distilbert import DistilBertTokenizer
from transformers.tokenization_roberta import RobertaTokenizer
from transformers.tokenization_xlnet import XLNetTokenizer

from utils_nlp.common.pytorch_utils import (
    get_device,
    move_model_to_device,
    get_amp,
    parallelize_model,
)

TOKENIZER_CLASS = {}
TOKENIZER_CLASS.update({k: BertTokenizer for k in BERT_PRETRAINED_MODEL_ARCHIVE_MAP})
TOKENIZER_CLASS.update(
    {k: RobertaTokenizer for k in ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP}
)
TOKENIZER_CLASS.update({k: XLNetTokenizer for k in XLNET_PRETRAINED_MODEL_ARCHIVE_MAP})
TOKENIZER_CLASS.update(
    {k: DistilBertTokenizer for k in DISTILBERT_PRETRAINED_MODEL_ARCHIVE_MAP}
)

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
                "names.".format(model_name, self.__class__.__name__)
            )
        self._model_name = model_name
        self._model_type = model_name.split("-")[0]
        self.cache_dir = cache_dir
        self.load_model_from_dir = load_model_from_dir
        if load_model_from_dir is None:
            self.model = model_class[model_name].from_pretrained(
                model_name,
                cache_dir=cache_dir,
                num_labels=num_labels,
                output_loading_info=False,
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

    @staticmethod
    def get_default_optimizer(model, weight_decay, learning_rate, adam_epsilon):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon
        )
        return optimizer

    @staticmethod
    def get_default_scheduler(optimizer, warmup_steps, num_training_steps):
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
        )
        return scheduler

    def prepare_model_and_optimizer(
        self,
        num_gpus,
        gpu_ids,
        local_rank,
        weight_decay,
        learning_rate,
        adam_epsilon,
        fp16=False,
        fp16_opt_level="O1",
        checkpoint_state_dict=None,
    ):
        """
        This function initializes an optimizer and moves the model to a device.
        It can be used by most child classes before calling fine_tune.
        Child classes that require custom optimizers need to either override this
            function or implement the steps listed below in the specified order
            before fine-tuning.

        The steps are performed in the following order:
            1. Move model to device
            2. Create optimizer
            3. Initialize amp
            4. Parallelize model
        """

        amp = get_amp(fp16)

        # get device
        device, num_gpus = get_device(
            num_gpus=num_gpus, gpu_ids=gpu_ids, local_rank=local_rank
        )

        # move model
        self.model = move_model_to_device(model=self.model, device=device)

        # init optimizer
        self.optimizer = Transformer.get_default_optimizer(
            self.model, weight_decay, learning_rate, adam_epsilon
        )

        if fp16:
            self.model, self.optimizer = amp.initialize(
                self.model, self.optimizer, opt_level=fp16_opt_level
            )

        if checkpoint_state_dict:
            self.optimizer.load_state_dict(checkpoint_state_dict["optimizer"])
            self.model.load_state_dict(checkpoint_state_dict["model"])

            if fp16:
                amp.load_state_dict(checkpoint_state_dict["amp"])

        self.model = parallelize_model(
            model=self.model,
            device=device,
            num_gpus=num_gpus,
            gpu_ids=gpu_ids,
            local_rank=local_rank,
        )

        return device, num_gpus, amp

    def fine_tune(
        self,
        train_dataloader,
        get_inputs,
        device,
        num_gpus=None,
        max_steps=-1,
        global_step=0,
        max_grad_norm=1.0,
        gradient_accumulation_steps=1,
        optimizer=None,
        scheduler=None,
        fp16=False,
        amp=None,
        local_rank=-1,
        verbose=True,
        seed=None,
        report_every=10,
        save_every=-1,
        clip_grad_norm=True,
    ):

        if seed is not None:
            Transformer.set_seed(seed, num_gpus > 0)

        # init training
        tr_loss = 0.0
        accum_loss = 0
        self.model.train()
        self.model.zero_grad()

        # train
        start = time.time()
        while global_step < max_steps:
            epoch_iterator = tqdm(
                train_dataloader,
                desc="Iteration",
                disable=local_rank not in [-1, 0] or not verbose,
            )
            for step, batch in enumerate(epoch_iterator):
                inputs = get_inputs(batch, device, self.model_name)
                outputs = self.model(**inputs)
                loss = outputs[0]

                if num_gpus > 1:
                    loss = loss.mean()
                if gradient_accumulation_steps > 1:
                    loss = loss / gradient_accumulation_steps

                if fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                tr_loss += loss.item()
                accum_loss += loss.item()

                if (step + 1) % gradient_accumulation_steps == 0:
                    global_step += 1

                    if clip_grad_norm:
                        if fp16:
                            torch.nn.utils.clip_grad_norm_(
                                amp.master_params(optimizer), max_grad_norm
                            )
                        else:
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(), max_grad_norm
                            )

                    if global_step % report_every == 0 and verbose:
                        end = time.time()
                        endtime_string = datetime.datetime.fromtimestamp(end).strftime(
                            "%d/%m/%Y %H:%M:%S"
                        )
                        print(
                            """timestamp: {0:s}, loss: {1:.6f}, time duration: {2:f},
                            number of examples in current step: {3:.0f}, step {4:.0f}
                            out of total {5:.0f}""".format(
                                endtime_string,
                                accum_loss / report_every,
                                end - start,
                                len(batch),
                                global_step,
                                max_steps,
                            )
                        )
                        accum_loss = 0
                        start = end

                    optimizer.step()
                    if scheduler:
                        scheduler.step()
                    self.model.zero_grad()
                    if save_every != -1 and global_step % save_every == 0 and verbose:
                        self.save_model(
                            os.path.join(
                                self.cache_dir,
                                f"{self.model_name}_step_{global_step}.pt",
                            )
                        )
                if global_step > max_steps:
                    epoch_iterator.close()
                    break

        return global_step, tr_loss / global_step

    def predict(self, eval_dataloader, get_inputs, num_gpus, gpu_ids, verbose=True):
        # get device
        device, num_gpus = get_device(num_gpus=num_gpus, gpu_ids=gpu_ids, local_rank=-1)

        # move model
        self.model = move_model_to_device(model=self.model, device=device)

        # parallelize model
        self.model = parallelize_model(
            model=self.model,
            device=device,
            num_gpus=num_gpus,
            gpu_ids=gpu_ids,
            local_rank=-1,
        )

        # predict
        self.model.eval()
        for batch in tqdm(eval_dataloader, desc="Scoring", disable=not verbose):
            with torch.no_grad():
                inputs = get_inputs(batch, device, self.model_name, train_mode=False)
                outputs = self.model(**inputs)
                logits = outputs[0]
            yield logits.detach().cpu().numpy()

    def save_model(self, full_name=None):
        """
        save the trained model.

        Args:
            full_name (str, optional): File name to save the model's `state_dict()`
                that can be loaded by torch.load().
                If None, the trained model, configuration and tokenizer are saved
                using `save_pretrained()`; and the file is going to be saved under
                "fine_tuned" folder of the cached directory of the object.
                Defaults to None.
        """

        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = (
            self.model.module if hasattr(self.model, "module") else self.model
        )  # Take care of distributed/parallel training

        if full_name:
            logger.info("Saving model checkpoint to %s", full_name)
            torch.save(model_to_save.state_dict(), full_name)
        else:
            output_model_dir = os.path.join(self.cache_dir, "fine_tuned")

            os.makedirs(self.cache_dir, exist_ok=True)
            os.makedirs(output_model_dir, exist_ok=True)

            logger.info("Saving model checkpoint to %s", output_model_dir)
            model_to_save.save_pretrained(output_model_dir)
