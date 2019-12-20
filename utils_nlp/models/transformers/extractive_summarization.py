# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# This script reuses some code from https://github.com/nlpyang/BertSum

import functools
import itertools
import logging
import numpy as np
import os
import random
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, IterableDataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import DistilBertModel, BertModel

from bertsum.models import model_builder, data_loader
from bertsum.models.data_loader import Batch, DataIterator
from bertsum.models.model_builder import Summarizer

from utils_nlp.common.pytorch_utils import get_device
from utils_nlp.models.transformers.common import MAX_SEQ_LEN, TOKENIZER_CLASS, Transformer
from utils_nlp.dataset.sentence_selection import combination_selection, greedy_selection

MODEL_CLASS = {"bert-base-uncased": BertModel, "distilbert-base-uncased": DistilBertModel}

logger = logging.getLogger(__name__)


class Bunch(object):
    """ Class which convert a dictionary to an object """

    def __init__(self, adict):
        self.__dict__.update(adict)


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

class TransformerSumData():
    def __init__(self, args, tokenizer):
        self.args = args
        self.tokenizer = tokenizer
        self.sep_vid = self.tokenizer.vocab['[SEP]']
        self.cls_vid = self.tokenizer.vocab['[CLS]']
        self.pad_vid = self.tokenizer.vocab['[PAD]']


    def preprocess(self, src, tgt=None, oracle_ids=None):

        if (len(src) == 0):
            return None

        original_src_txt = [' '.join(s) for s in src]

        labels = None
        if oracle_ids is not None and tgt is not None:
            labels = [0] * len(src)
            for l in oracle_ids:
                labels[l] = 1

        idxs = [i for i, s in enumerate(src) if (len(s) > self.args.min_src_ntokens)]

        src = [src[i][:self.args.max_src_ntokens] for i in idxs]
        src = src[:self.args.max_nsents]
        if labels:
            labels = [labels[i] for i in idxs]
            labels = labels[:self.args.max_nsents]

        if (len(src) < self.args.min_nsents):
            return None
        if labels:
            if (len(labels) == 0):
                return None

        src_txt = [' '.join(sent) for sent in src]
        # text = [' '.join(ex['src_txt'][i].split()[:self.args.max_src_ntokens]) for i in idxs]
        # text = [_clean(t) for t in text]
        text = ' [SEP] [CLS] '.join(src_txt)
        src_subtokens = self.tokenizer.tokenize(text)
        src_subtokens = src_subtokens[:510]
        src_subtokens = ['[CLS]'] + src_subtokens + ['[SEP]']

        src_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(src_subtokens)
        _segs = [-1] + [i for i, t in enumerate(src_subtoken_idxs) if t == self.sep_vid]
        segs = [_segs[i] - _segs[i - 1] for i in range(1, len(_segs))]
        segments_ids = []
        for i, s in enumerate(segs):
            if (i % 2 == 0):
                segments_ids += s * [0]
            else:
                segments_ids += s * [1]
        cls_ids = [i for i, t in enumerate(src_subtoken_idxs) if t == self.cls_vid]
        if labels:
            labels = labels[:len(cls_ids)]

        tgt_txt = None
        if tgt:
            tgt_txt = '<q>'.join([' '.join(tt) for tt in tgt])
        src_txt = [original_src_txt[i] for i in idxs]
        return src_subtoken_idxs, labels, segments_ids, cls_ids, src_txt, tgt_txt

def get_dataset(file):
    yield torch.load(file)
            
class ExmSumProcessedIterableDataset(IterableDataset):
    
    def __init__(self, file_list, is_shuffle=False):
        self.file_list = file_list
        self.is_shuffle = is_shuffle
        
    def get_stream(self):
        if self.is_shuffle:
            return itertools.chain.from_iterable(map(get_dataset, itertools.cycle(self.file_list)))
        else:
            return itertools.chain.from_iterable(map(get_dataset, itertools.cycle(random.shuffle(self.file_list))))
                                             
    def __iter__(self):
        return self.get_stream()
                                            
class ExmSumProcessedDataset(Dataset):
    
    def __init__(self, file_list, is_shuffle=False):
        self.file_list = file_list
        if is_shuffle:
            random.shuffle(file_list)
        self.data = []
        for file in file_list:
            self.data.extend(torch.load(file))
    
    def __len__(self):
        return len(self.data)
                                                     
    def __getitem__(self, idx):
        return self.data[idx]

    
def get_pred(example, sent_scores, cal_lead=False, sentence_seperator='<q>', block_trigram=True, top_n=3):
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

    selected_ids = np.argsort(-sent_scores)
    #selected_ids = np.argsort(-sent_scores, 1)
    if cal_lead:
        selected_ids = range(len(example['clss']))
    pred = []
    #target = []
    #for i, idx in enumerate(selected_ids):
    _pred = []
    if len(example['src_txt']) == 0:
        pred.append("")
    for j in selected_ids[: len(example['src_txt'])]:
        if j >= len(example['src_txt']):
            continue
        candidate = example['src_txt'][j].strip()
        if block_trigram:
            if not _block_tri(candidate, _pred):
                _pred.append(candidate)
        else:
            _pred.append(candidate)

        # only select the top n
        if len(_pred) == top_n:
            break

    # _pred = '<q>'.join(_pred)
    _pred = sentence_seperator.join(_pred)
    pred.append(_pred.strip())
    #target.append(example['tgt_txt'])
    return pred #, target

class ExtSumProcessedData:
    @staticmethod
    def save_data(data_iter, is_test=False, save_path="./", chunk_size=None):
        os.makedirs(save_path, exist_ok=True)

        def chunks(iterable, chunk_size):
            iterator = filter(None, iterable)  
            for first in iterator:
                if chunk_size:
                    yield itertools.chain([first], itertools.islice(iterator, chunk_size - 1))
                else:
                    yield itertools.chain([first], itertools.islice(iterator, None))

        chunks = chunks(data_iter, chunk_size)
        filename_list = []
        for i, chunked_data in enumerate(chunks):
            filename = f"{i}_test" if is_test else f"{i}_train"
            torch.save(list(chunked_data), os.path.join(save_path, filename))
            filename_list.append(os.path.join(save_path, filename))
        return filename_list
    
   
    def get_files(self, root):
        train_files = []
        test_files = []
        files = [os.path.join(root, f) for f in os.listdir(root) if os.path.isfile(os.path.join(root, f))]
        for fname in files:
            if fname.find("train") != -1:
                train_files.append(fname)
            elif fname.find("test") != -1:
                test_files.append(fname)
                
        return train_files, test_files
    
 
    def splits(self, root):
        train_files, test_files = self.get_files(root)
        return ExmSumProcessedIterableDataset(train_files, is_shuffle=True), ExmSumProcessedDataset(test_files, is_shuffle=False)
        
        

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
        self.processor = TransformerSumData(args, self.tokenizer)

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

        if targets is None:
            for source in sources:
                yield self._preprocess_single(source, None, oracle_mode, selections)
        else:
            for (source, target) in zip(sources, targets):
                yield self._preprocess_single(source, target, oracle_mode, selections)

    def _preprocess_single(self, source, target=None, oracle_mode="greedy", selections=3):
        """preprocess single data point"""
        oracle_ids = None
        if target is not None:
            if oracle_mode == "greedy":
                oracle_ids = greedy_selection(source, target, selections)
            elif oracle_mode == "combination":
                oracle_ids = combination_selection(source, target, selections)

        b_data = self.processor.preprocess(source, target, oracle_ids)
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
        self.model = Summarizer(encoder, args, model_class, model_name, None, cache_dir)

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
        test_dataset,
        num_gpus=1,
        batch_size=16,
        sentence_seperator="<q>",
        top_n=3,
        block_trigram=True,
        verbose=True,
        cal_lead=False,
    ):
    
        def collate_fn(dict_list):
            # tuple_batch =  [list(col) for col in zip(*[d.values() for d in dict_list]
            if dict_list is None or len(dict_list) <= 0:
                return None
            is_labeled = False
            if "labels" in dict_list[0]:
                is_labeled = True
            tuple_batch = [list(d.values()) for d in dict_list]
            ## generate mask and mask_cls, and only select tensors for the model input
            batch = Batch(tuple_batch, is_labeled=True)    
            if is_labeled:
                return {
                    "src": batch.src,
                    "segs": batch.segs,
                    "clss": batch.clss,
                    "mask": batch.mask,
                    "mask_cls": batch.mask_cls,
                    "labels": batch.labels,
                }
            else:
                return {
                    "src": batch.src,
                    "segs": batch.segs,
                    "clss": batch.clss,
                    "mask": batch.mask,
                    "mask_cls": batch.mask_cls,
                }
        
        test_sampler = SequentialSampler(test_dataset)
        test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=batch_size, collate_fn=collate_fn)
        sent_scores = self.predict_scores(test_dataloader, num_gpus=num_gpus)
        sent_scores_list = list(sent_scores)
        scores_list = []
        for i in sent_scores_list:
            scores_list.extend(i)
        prediction = []
        for i in range(len(test_dataset)):
            temp_pred = get_pred(test_dataset[i], scores_list[i])
            prediction.extend(temp_pred)
            print(temp_pred[0])
            print(temp_target[0])
        return prediction
        
        
    def predict_scores(
        self,
        eval_dataloader,
        num_gpus=1,        
        verbose=True,
    ):

        device, num_gpus = get_device(num_gpus=num_gpus, local_rank=-1)
        # if isinstance(self.model, nn.DataParallel):
        #    self.model.module.to(device)
        # else:
        self.model.to(device)

        #def move_batch_to_device(batch, device):
        #    return batch.to(device)
        def move_batch_to_device(batch, device):
            batch['src'] = batch['src'].to(device)
            batch['segs'] = batch['segs'].to(device)
            batch['clss'] = batch['clss'].to(device)
            batch['mask'] = batch['mask'].to(device)
            batch['mask_cls'] = batch['mask_cls'].to(device)
            if 'labels' in batch:
                batch['labels'] = batch['labels'].to(device)
            return Bunch(batch)

        self.model.eval()

        for batch in eval_dataloader:
            batch = move_batch_to_device(batch, device)
            with torch.no_grad():
                inputs = ExtSumProcessor.get_inputs(batch, self.model_name, train_mode=False)
                outputs = self.model(**inputs)
                sent_scores = outputs[0]
                sent_scores = sent_scores.detach().cpu().numpy()
                yield sent_scores
               

    def save_model(self, name):
        output_model_dir = os.path.join(self.cache_dir, "fine_tuned")

        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(output_model_dir, exist_ok=True)

        full_name = os.path.join(output_model_dir, name)
        logger.info("Saving model checkpoint to %s", full_name)
        torch.save(self.model, name)
      