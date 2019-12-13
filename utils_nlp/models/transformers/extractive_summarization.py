# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import itertools
import logging
import numpy as np
import os
import random
import time
import torch
import torch.nn as nn

# from torch.utils.data import Dataset, TensorDataset
# from transformers import *


from transformers.modeling_bert import (
    BERT_PRETRAINED_MODEL_ARCHIVE_MAP,
    BertForSequenceClassification,
)
from transformers.modeling_distilbert import (
    DISTILBERT_PRETRAINED_MODEL_ARCHIVE_MAP,
    DistilBertForSequenceClassification,
)
from transformers.modeling_roberta import (
    ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP,
    RobertaForSequenceClassification,
)
from transformers.modeling_xlnet import (
    XLNET_PRETRAINED_MODEL_ARCHIVE_MAP,
    XLNetForSequenceClassification,
)

from transformers import DistilBertModel, BertModel


from utils_nlp.common.pytorch_utils import get_device
from utils_nlp.models.transformers.common import MAX_SEQ_LEN, TOKENIZER_CLASS, Transformer


from bertsum.models import model_builder, data_loader
from bertsum.models.data_loader import DataIterator
from bertsum.models.model_builder import Summarizer
from bertsum.prepro.data_builder import combination_selection, greedy_selection, TransformerData

MODEL_CLASS = {"bert-base-uncased": BertModel, "distilbert-base-uncased": DistilBertModel}

logger = logging.getLogger(__name__)


class Bunch(object):
    """ Class which convert a dictionary to an object """

    def __init__(self, adict):
        self.__dict__.update(adict)


def get_sequential_dataloader(dataset, is_labeled=False, batch_size=3000):
    """
    Function to get sequential data iterator over a list of data objects.
    Args:
        dataset (list of objects): a list of data objects.
        is_test (bool): it specifies whether the data objects are labeled data.
        batch_size (int): number of tokens per batch.
        
    Returns:
        DataIterator
    """

    return DataIterator(dataset, batch_size, is_labeled=is_labeled, shuffle=False, sort=False)


def get_cycled_dataset(train_dataset_generator):
    """
    Function to get iterator over the dataset specified by train_iter.
    It cycles through the dataset.
    """
    cycle_iterator = itertools.cycle("123")
    for _ in cycle_iterator:
        for batch in train_dataset_generator():
            yield batch


def get_dataset(file_list, is_train=False):
    if is_train:
        random.shuffle(file_list)
    for file in file_list:
        yield torch.load(file)


def get_dataloader(data_iter, shuffle=True, is_labeled=False, batch_size=3000):
    """
    Function to get data iterator over a list of data objects.

    Args:
        data_iter (generator): data generator.
        shuffle (bool): whether the data is shuffled
        is_labeled (bool): it specifies whether the data objects are labeled data.
        batch_size (int): number of tokens per batch.
        
    Returns:
        DataIterator

    """

    return data_loader.Dataloader(data_iter, batch_size, shuffle=shuffle, is_labeled=is_labeled)


class ExtSumProcessor:
    def __init__(
        self,
        model_name="bert-base-cased",
        to_lower=False,
        cache_dir=".",
        max_nsents=200,
        max_src_ntokens=2000,
        min_nsents=3,
        min_src_ntokens=5,
        use_interval=True,
    ):
        self.tokenizer = TOKENIZER_CLASS[model_name].from_pretrained(
            model_name, do_lower_case=to_lower, cache_dir=cache_dir
        )

        default_preprocessing_parameters = {
            "max_nsents": max_nsents,
            "max_src_ntokens": max_src_ntokens,
            "min_nsents": min_nsents,
            "min_src_ntokens": min_src_ntokens,
            "use_interval": use_interval,
        }
        print(default_preprocessing_parameters)
        args = Bunch(default_preprocessing_parameters)
        self.preprossor = TransformerData(args, self.tokenizer)

    @staticmethod
    def get_inputs(batch, model_name, train_mode=True):
        if model_name.split("-")[0] in ["bert", "distilbert"]:
            if train_mode:
                # labels must be the last
                return {
                    "x": batch.src,
                    "segs": batch.segs,
                    "clss": batch.clss,
                    "mask": batch.mask,
                    "mask_cls": batch.mask_cls,
                    "labels": batch.labels,
                }
            else:
                return {
                    "x": batch.src,
                    "segs": batch.segs,
                    "clss": batch.clss,
                    "mask": batch.mask,
                    "mask_cls": batch.mask_cls,
                }
        else:
            raise ValueError("Model not supported: {}".format(model_name))

    def preprocess(self, sources, targets=None, oracle_mode="greedy", selections=3):
        """preprocess multiple data points"""

        is_labeled = False
        if targets is None:
            for source in sources:
                yield self._preprocess_single(source, None, oracle_mode, selections)
        else:
            for (source, target) in zip(sources, targets):
                yield self._preprocess_single(source, target, oracle_mode, selections)
            is_labeled = True

    def _preprocess_single(self, source, target=None, oracle_mode="greedy", selections=3):
        """preprocess single data point"""
        oracle_ids = None
        if target is not None:
            if oracle_mode == "greedy":
                oracle_ids = greedy_selection(source, target, selections)
            elif oracle_mode == "combination":
                oracle_ids = combination_selection(source, target, selections)

        b_data = self.preprossor.preprocess(source, target, oracle_ids)
        if b_data is None:
            return None
        indexed_tokens, labels, segments_ids, cls_ids, src_txt, tgt_txt = b_data
        return {
            "src": indexed_tokens,
            "labels": labels,
            "segs": segments_ids,
            "clss": cls_ids,
            "src_txt": src_txt,
            "tgt_txt": tgt_txt,
        }


class ExtractiveSummarizer(Transformer):
    def __init__(self, model_name="distilbert-base-uncased", encoder="transformer", cache_dir="."):
        super().__init__(
            model_class=MODEL_CLASS, model_name=model_name, num_labels=0, cache_dir=cache_dir
        )
        model_class = MODEL_CLASS[model_name]
        default_summarizer_layer_parameters = {
            "ff_size": 512,
            "heads": 4,
            "dropout": 0.1,
            "inter_layers": 2,
            "hidden_size": 128,
            "rnn_size": 512,
            "param_init": 0.0,
            "param_init_glorot": True,
        }

        args = Bunch(default_summarizer_layer_parameters)
        self.model = Summarizer("transformer", args, model_class, model_name, None, cache_dir)

    @staticmethod
    def list_supported_models():
        return list(MODEL_CLASS)

    def fit(
        self,
        train_dataloader,
        num_gpus=None,
        local_rank=-1,
        max_steps=5e5,
        optimization_method="adam",
        lr=2e-3,
        max_grad_norm=0,
        beta1=0.9,
        beta2=0.999,
        decay_method="noam",
        warmup_steps=1e5,
        verbose=True,
        seed=None,
        gradient_accumulation_steps=2,
        report_every=50,
        clip_grad_norm=False,
        **kwargs
    ):

        device, num_gpus = get_device(num_gpus=num_gpus, local_rank=local_rank)

        def move_batch_to_device(batch, device):
            return batch.to(device)

        # if isinstance(self.model, nn.DataParallel):
        #    self.model.module.to(device)
        # else:
        self.model.to(device)

        optimizer = model_builder.build_optim(
            optimization_method,
            lr,
            max_grad_norm,
            beta1,
            beta2,
            decay_method,
            warmup_steps,
            self.model,
            None,
        )

        # train_dataloader = get_dataloader(train_iter(), is_labeled=True, batch_size=batch_size)

        super().fine_tune(
            train_dataloader=train_dataloader,
            get_inputs=ExtSumProcessor.get_inputs,
            device=device,
            move_batch_to_device=move_batch_to_device,
            n_gpu=num_gpus,
            num_train_epochs=-1,
            max_steps=max_steps,
            optimizer=optimizer,
            warmup_steps=warmup_steps,
            gradient_accumulation_steps=gradient_accumulation_steps,
            verbose=verbose,
            seed=seed,
            report_every=report_every,
            clip_grad_norm=clip_grad_norm,
            max_grad_norm=max_grad_norm,
        )

    def predict(
        self,
        eval_dataloader,
        num_gpus=1,
        batch_size=16,
        sentence_seperator="<q>",
        top_n=3,
        block_trigram=True,
        verbose=True,
        cal_lead=False,
    ):
        def _get_ngrams(n, text):
            ngram_set = set()
            text_length = len(text)
            max_index_ngram_start = text_length - n
            for i in range(max_index_ngram_start + 1):
                ngram_set.add(tuple(text[i : i + n]))
            return ngram_set

        def _block_tri(c, p):
            tri_c = _get_ngrams(3, c.split())
            for s in p:
                tri_s = _get_ngrams(3, s.split())
                if len(tri_c.intersection(tri_s)) > 0:
                    return True
            return False

        def _get_pred(batch, sent_scores):
            # return sent_scores
            if cal_lead:
                selected_ids = list(range(batch.clss.size(1))) * len(batch.clss)
            else:
                # negative_sent_score = [-i for i in sent_scores[0]]
                selected_ids = np.argsort(-sent_scores, 1)
                # selected_ids = np.sort(selected_ids,1)
            pred = []
            for i, idx in enumerate(selected_ids):
                _pred = []
                if len(batch.src_str[i]) == 0:
                    pred.append("")
                    continue
                for j in selected_ids[i][: len(batch.src_str[i])]:
                    if j >= len(batch.src_str[i]):
                        continue
                    candidate = batch.src_str[i][j].strip()
                    if block_trigram:
                        if not _block_tri(candidate, _pred):
                            _pred.append(candidate)
                    else:
                        _pred.append(candidate)

                    # only select the top 3
                    if len(_pred) == top_n:
                        break

                # _pred = '<q>'.join(_pred)
                _pred = sentence_seperator.join(_pred)
                pred.append(_pred.strip())
            return pred

        device, num_gpus = get_device(num_gpus=num_gpus, local_rank=-1)
        # if isinstance(self.model, nn.DataParallel):
        #    self.model.module.to(device)
        # else:
        self.model.to(device)

        def move_batch_to_device(batch, device):
            return batch.to(device)

        self.model.eval()
        pred = []
        for batch in eval_dataloader:
            batch = move_batch_to_device(batch, device)
            with torch.no_grad():
                inputs = ExtSumProcessor.get_inputs(batch, self.model_name, train_mode=False)
                outputs = self.model(**inputs)
                sent_scores = outputs[0]
                sent_scores = sent_scores.detach().cpu().numpy()
                pred.extend(_get_pred(batch, sent_scores))
        return pred

    def save_model(self, name):
        output_model_dir = os.path.join(self.cache_dir, "fine_tuned")

        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(output_model_dir, exist_ok=True)

        full_name = os.path.join(output_model_dir, name)
        logger.info("Saving model checkpoint to %s", full_name)
        torch.save(self.model, name)
