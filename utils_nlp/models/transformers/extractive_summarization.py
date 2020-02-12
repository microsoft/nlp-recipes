# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# This script reuses some code from https://github.com/nlpyang/BertSum

import gc
import itertools
import logging
import os
import pickle
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset, SequentialSampler

from transformers import BertModel, DistilBertModel

from bertsum.models import model_builder
from bertsum.models.data_loader import Batch, DataIterator
from bertsum.models.model_builder import Summarizer
from utils_nlp.common.pytorch_utils import (
    compute_training_steps,
    get_device,
    move_model_to_device,
    parallelize_model,
)
from utils_nlp.dataset.sentence_selection import combination_selection, greedy_selection
from utils_nlp.models.transformers.common import TOKENIZER_CLASS, Transformer

MODEL_CLASS = {
    "bert-base-uncased": BertModel,
    "distilbert-base-uncased": DistilBertModel,
}

logger = logging.getLogger(__name__)


# https://pytorch.org/docs/1.1.0/_modules/torch/utils/data/dataloader.html
class IterableDistributedSampler(object):
    """ Distributed sampler for iterable dataset.

    Args:
        world_size (int): Total number of GPUs that will be used. Defaults to 1.
        rank (int): Rank of the current GPU. Defaults to -1.

    """

    def __init__(self, world_size=1, rank=-1):
        self.world_size = world_size
        self.rank = rank

    def iter(self, iterable):
        if self.rank != -1:
            return itertools.islice(iterable, self.rank, None, self.world_size)
        else:
            return iterable


class ChunkDataLoader(object):
    """ Data Loader for Chunked Dataset.

    Args:
        datasets (list): list of data item list.
        batch_size (int): Number of tokens per batch.
        shuffle (bool): Whether the data is shuffled.
        is_labeled (bool): Whether the data is labeled.
        sampler (obj): Data sampler.

    """

    def __init__(self, datasets, batch_size, shuffle, is_labeled, sampler):
        self.datasets = datasets
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.is_labeled = is_labeled
        self.cur_iter = self._next_dataset_iterator(datasets)
        assert self.cur_iter is not None
        self.sampler = sampler

    def eachiter(self):
        dataset_iter = (d for d in self.datasets)
        while self.cur_iter is not None:
            for batch in self.cur_iter:
                yield batch
            self.cur_iter = self._next_dataset_iterator(dataset_iter)

    def __iter__(self):
        return self.sampler.iter(self.eachiter())

    def _next_dataset_iterator(self, dataset_iter):
        try:
            # Drop the current dataset for decreasing memory
            if hasattr(self, "cur_dataset"):
                self.cur_dataset = None
                gc.collect()
                del self.cur_dataset
                gc.collect()

            self.cur_dataset = next(dataset_iter)
        except StopIteration:
            return None

        return DataIterator(
            dataset=self.cur_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            is_labeled=self.is_labeled,
        )


class Bunch(object):
    """ Class which convert a dictionary to an object """

    def __init__(self, adict):
        self.__dict__.update(adict)


def get_dataloader(
    data_iter, shuffle=True, is_labeled=False, batch_size=3000, world_size=1, rank=-1
):
    """
    Function to get data iterator over a list of data objects.

    Args:
        data_iter (generator): Data generator.
        shuffle (bool): Whether the data is shuffled. Defaults to True.
        is_labeled (bool): Whether the data objects are labeled data.
                            Defaults to False.
        batch_size (int): Number of tokens per batch. Defaults to 3000.
        world_size (int): Total number of GPUs that will be used. Defaults to 1.
        rank (int): Rank of the current GPU. Defaults to -1.

    Returns:
        DataIterator
    """
    sampler = IterableDistributedSampler(world_size, rank)
    return ChunkDataLoader(
        data_iter, batch_size, shuffle=shuffle, is_labeled=is_labeled, sampler=sampler
    )


def get_dataset(file):
    yield torch.load(file)


class ExtSumProcessedIterableDataset(IterableDataset):
    """Iterable dataset for extractive summarization preprocessed data
    """

    def __init__(self, file_list, is_shuffle=False):
        """ Initiation function for iterable dataset for extractive summarization
            preprocessed data.

        Args:
            file_list (list of strings): List of files that the dataset is loaded from.
            is_shuffle (bool, optional): A boolean value specifies whether the list of
                files is shuffled when the dataset is loaded. Defaults to False.
        """

        self.file_list = file_list
        self.is_shuffle = is_shuffle

    def get_stream(self):
        """ get a stream of cycled data from the dataset"""

        if self.is_shuffle:
            return itertools.chain.from_iterable(
                map(get_dataset, itertools.cycle(self.file_list))
            )
        else:
            return itertools.chain.from_iterable(
                map(get_dataset, itertools.cycle(random.shuffle(self.file_list)))
            )

    def __iter__(self):
        return self.get_stream()


class ExtSumProcessedDataset(Dataset):
    """Dataset for extractive summarization preprocessed data
    """

    def __init__(self, file_list, is_shuffle=False):
        """ Initiation function for dataset for extractive summarization preprocessed data.

        Args:
            file_list (list of strings): List of files that the dataset is loaded from.
            is_shuffle (bool, optional): A boolean value specifies whether the list of
                files is shuffled when the dataset is loaded. Defaults to False.
        """

        self.file_list = sorted(file_list)
        if is_shuffle:
            random.shuffle(self.file_list)
        self.data = []
        for f in self.file_list:
            self.data.extend(torch.load(f))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def get_pred(
    example,
    sent_scores,
    cal_lead=False,
    sentence_separator="<q>",
    block_trigram=True,
    top_n=3,
):
    """
        Get the summarization prediction for the paragraph example based on the scores
        returned by the transformer summarization model.

        Args:
            example (str): The object with "src_txt" field as the paragraph which
                requries summarization. The "src_txt" is a list of strings.
            sent_scores (list of floats): List of scores of how likely of the
                sentence is included in the summary.
            cal_lead (bool, optional): Boolean value which specifies whether the
                prediction uses the first few sentences as summary. Defaults to False.
            sentence_separator (str, optional): Seperator used in the generated summary.
                Defaults to '<q>'.
            block_trigram (bool, optional): Boolean value which specifies whether the
                summary should include any sentence that has the same trigram as the
                already selected sentences. Defaults to True.
            top_n (int, optional): The maximum number of sentences that the summary
                should included. Defaults to 3.

        Returns:
            A string which is the summary for the example.
    """

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
    # selected_ids = np.argsort(-sent_scores, 1)
    if cal_lead:
        selected_ids = range(len(example["clss"]))
    pred = []
    # target = []
    # for i, idx in enumerate(selected_ids):
    _pred = []
    if len(example["src_txt"]) == 0:
        pred.append("")
    for j in selected_ids[: len(example["src_txt"])]:
        if j >= len(example["src_txt"]):
            continue
        candidate = example["src_txt"][j].strip()
        if block_trigram:
            if not _block_tri(candidate, _pred):
                _pred.append(candidate)
        else:
            _pred.append(candidate)

        # only select the top n
        if len(_pred) == top_n:
            break

    # _pred = '<q>'.join(_pred)
    _pred = sentence_separator.join(_pred)
    pred.append(_pred.strip())
    # target.append(example['tgt_txt'])
    return pred  # , target


class ExtSumProcessedData:
    """class loaded data preprocessed as in
    :class:`utils_nlp.models.transformers.datasets.SummarizationDataset`"""

    @staticmethod
    def save_data(data_iter, is_test=False, save_path="./", chunk_size=None):
        """ Save the preprocessed data into files with specified chunk size

        Args:
            data_iter (iterator): Data iterator returned from
                :class:`utils_nlp.models.transformers.datasets.SummarizationDataset`
            is_test (bool): Boolean value which indicates whether target data
                is included. If set to True, the file name contains "test", otherwise,
                the file name contains "train". Defaults to False.
            save_path (str): Directory where the data should be saved. Defaults to "./".
            chunk_size (int): The number of examples that should be included in each
                file. Defaults to None, which means only one file is used.

        Returns:
            a list of strings which are the files the data is saved to.
        """
        os.makedirs(save_path, exist_ok=True)

        def _chunks(iterable, chunk_size):
            iterator = filter(None, iterable)
            for first in iterator:
                if chunk_size:
                    yield itertools.chain(
                        [first], itertools.islice(iterator, chunk_size - 1)
                    )
                else:
                    yield itertools.chain([first], itertools.islice(iterator, None))

        chunks = _chunks(data_iter, chunk_size)
        filename_list = []
        for i, chunked_data in enumerate(chunks):
            filename = f"{i}_test" if is_test else f"{i}_train"
            torch.save(list(chunked_data), os.path.join(save_path, filename))
            filename_list.append(os.path.join(save_path, filename))
        return filename_list

    def _get_files(self, root):
        train_files = []
        test_files = []
        files = [
            os.path.join(root, f)
            for f in os.listdir(root)
            if os.path.isfile(os.path.join(root, f))
        ]
        for fname in files:
            if fname.find("train") != -1:
                train_files.append(fname)
            elif fname.find("test") != -1:
                test_files.append(fname)

        return train_files, test_files

    def splits(self, root):
        """Get the train and test dataset from the folder

        Args:
            root (str): Directory where the data can be loaded.

        Returns:
            Tuple of ExtSumProcessedIterableDataset as train dataset
            and ExtSumProcessedDataset as test dataset.
        """
        train_files, test_files = self._get_files(root)
        return (
            ExtSumProcessedIterableDataset(train_files, is_shuffle=True),
            ExtSumProcessedDataset(test_files, is_shuffle=False),
        )


class ExtSumProcessor:
    """Class for preprocessing extractive summarization data."""

    def __init__(
        self,
        model_name="distilbert-base-uncased",
        to_lower=False,
        cache_dir=".",
        max_nsents=200,
        max_src_ntokens=2000,
        min_nsents=3,
        min_src_ntokens=5,
    ):
        """ Initialize the preprocessor.

        Args:
            model_name (str, optional): Transformer model name used in preprocessing.
                check MODEL_CLASS for supported models. Defaults to "bert-base-cased".
            to_lower (bool, optional): Whether to convert all letters to lower case
                during tokenization. This is determined by if a cased model is used.
                Defaults to False, which corresponds to a cased model.
            cache_dir (str, optional): Directory to cache the tokenizer.
                Defaults to ".".
            max_nsents (int, optional): Max number of sentences that can be used
                as input. Defaults to 200.
            max_src_ntokens (int, optional): Max number of tokens that be used
                as input. Defaults to 2000.
            min_nsents (int, optional): Minimum number of sentences that are required
                as input. If the input has less number of sentences than this value,
                it's skipped and cannot be used as a valid input. Defaults to 3.
            min_src_ntokens (int, optional): Minimum number of tokens that are required
                as an input sentence.If the input sentence has less number of tokens
                than this value, it's skipped and cannot be used as a valid sentence.
                Defaults to 5.

        """
        self.model_name = model_name
        self.tokenizer = TOKENIZER_CLASS[self.model_name].from_pretrained(
            self.model_name, do_lower_case=to_lower, cache_dir=cache_dir
        )
        self.sep_vid = self.tokenizer.vocab["[SEP]"]
        self.cls_vid = self.tokenizer.vocab["[CLS]"]
        self.pad_vid = self.tokenizer.vocab["[PAD]"]

        self.max_nsents = max_nsents
        self.max_src_ntokens = max_src_ntokens
        self.min_nsents = min_nsents
        self.min_src_ntokens = min_src_ntokens

    @staticmethod
    def list_supported_models():
        return list(TOKENIZER_CLASS.keys())

    @property
    def model_name(self):
        return self._model_name

    @model_name.setter
    def model_name(self, value):
        if value not in self.list_supported_models():
            raise ValueError(
                "Model name {} is not supported by ExtSumProcessor. "
                "Call 'ExtSumProcessor.list_supported_models()' to get all supported "
                "model names.".format(value)
            )

        self._model_name = value

    @staticmethod
    def get_inputs(batch, device, model_name, train_mode=True):
        """
        Creates an input dictionary given a model name.

        Args:
            batch (object): A Batch containing input ids, segment ids, sentence class
                ids, masks for the input ids, masks for  sentence class ids and source
                text. If train_model is True, it also contains the labels and target
                text.
            device (torch.device): A PyTorch device.
            model_name (bool, optional): Model name used to format the inputs.
            train_mode (bool, optional): Training mode flag.
                Defaults to True.

        Returns:
            dict: Dictionary containing input ids, segment ids, sentence class ids,
            masks for the input ids, masks for the sentence class ids and labels.
            Labels are only returned when train_mode is True.
        """

        if model_name.split("-")[0] in ["bert", "distilbert"]:
            if train_mode:
                batch = batch.to(device)
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
                batch = Bunch(batch)
                return {
                    "x": batch.src.to(device),
                    "segs": batch.segs.to(device),
                    "clss": batch.clss.to(device),
                    "mask": batch.mask.to(device),
                    "mask_cls": batch.mask_cls.to(device),
                }
        else:
            raise ValueError("Model not supported: {}".format(model_name))

    def preprocess(self, sources, targets=None, oracle_mode="greedy", selections=3):
        """preprocess multiple data points

           Args:
              sources (list of list of strings): List of word tokenized sentences.
              targets (list of list of strings, optional): List of word tokenized
                sentences.
                Defaults to None, which means it doesn't include summary and is
                not training data.
              oracle_mode (str, optional): Sentence selection method.
                Defaults to "greedy".
              selections (int, optional): The number of sentence used as summary.
                Defaults to 3.

            Returns:
                Iterator of dictory objects containing input ids, segment ids,
                sentence class ids, labels, source text and target text.
                If targets is None, the label and target text are None.
        """

        if targets is None:
            for source in sources:
                yield self._preprocess_single(source, None, oracle_mode, selections)
        else:
            for (source, target) in zip(sources, targets):
                yield self._preprocess_single(source, target, oracle_mode, selections)

    def _preprocess_single(
        self, source, target=None, oracle_mode="greedy", selections=3
    ):
        """preprocess single data point"""

        oracle_ids = None
        if target is not None:
            if oracle_mode == "greedy":
                oracle_ids = greedy_selection(source, target, selections)
            elif oracle_mode == "combination":
                oracle_ids = combination_selection(source, target, selections)

        def _preprocess(src, tgt=None, oracle_ids=None):

            if len(src) == 0:
                return None

            original_src_txt = [" ".join(s) for s in src]

            labels = None
            if oracle_ids is not None and tgt is not None:
                labels = [0] * len(src)
                for l in oracle_ids:
                    labels[l] = 1

            idxs = [i for i, s in enumerate(src) if (len(s) > self.min_src_ntokens)]

            src = [src[i][: self.max_src_ntokens] for i in idxs]
            src = src[: self.max_nsents]
            if labels:
                labels = [labels[i] for i in idxs]
                labels = labels[: self.max_nsents]

            if len(src) < self.min_nsents:
                return None
            if labels:
                if len(labels) == 0:
                    return None

            src_txt = [" ".join(sent) for sent in src]
            # text = [' '.join(ex['src_txt'][i].split()[:self.args.max_src_ntokens])
            #  for i in idxs]
            # text = [_clean(t) for t in text]
            text = " [SEP] [CLS] ".join(src_txt)
            src_subtokens = self.tokenizer.tokenize(text)
            src_subtokens = src_subtokens[:510]
            src_subtokens = ["[CLS]"] + src_subtokens + ["[SEP]"]

            src_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(src_subtokens)
            _segs = [-1] + [
                i for i, t in enumerate(src_subtoken_idxs) if t == self.sep_vid
            ]
            segs = [_segs[i] - _segs[i - 1] for i in range(1, len(_segs))]
            segments_ids = []
            for i, s in enumerate(segs):
                if i % 2 == 0:
                    segments_ids += s * [0]
                else:
                    segments_ids += s * [1]
            cls_ids = [i for i, t in enumerate(src_subtoken_idxs) if t == self.cls_vid]
            if labels:
                labels = labels[: len(cls_ids)]

            tgt_txt = None
            if tgt:
                tgt_txt = "<q>".join([" ".join(tt) for tt in tgt])
            src_txt = [original_src_txt[i] for i in idxs]
            return src_subtoken_idxs, labels, segments_ids, cls_ids, src_txt, tgt_txt

        b_data = _preprocess(source, target, oracle_ids)

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
    """class which performs extractive summarization fine tuning and prediction """

    def __init__(
        self, model_name="distilbert-base-uncased", encoder="transformer", cache_dir="."
    ):
        """Initialize a ExtractiveSummarizer.

        Args:
            model_name (str, optional): Transformer model name used in preprocessing.
                check MODEL_CLASS for supported models.
                Defaults to "distilbert-base-uncased".
            encoder (str, optional): Encoder algorithm used by summarization layer.
                There are four options:
                    - baseline: it used a smaller transformer model to replace the bert
                        model and with transformer summarization layer.
                    - classifier: it uses pretrained BERT and fine-tune BERT with simple
                        logistic classification summarization layer.
                    - transformer: it uses pretrained BERT and fine-tune BERT with
                        transformer summarization layer.
                    - RNN: it uses pretrained BERT and fine-tune BERT with LSTM
                        summarization layer.
                Defaults to "transformer".
            cache_dir (str, optional): Directory to cache the tokenizer.
                Defaults to ".".
        """

        super().__init__(
            model_class=MODEL_CLASS,
            model_name=model_name,
            num_labels=0,
            cache_dir=cache_dir,
        )
        if model_name not in self.list_supported_models():
            raise ValueError(
                "Model name {} is not supported by ExtractiveSummarizer. "
                "Call 'ExtractiveSummarizer.list_supported_models()' to get all  "
                "supported model names.".format(model_name)
            )

        self.model_class = MODEL_CLASS[model_name]
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
        self.model = Summarizer(
            encoder, args, self.model_class, model_name, None, cache_dir
        )

    @staticmethod
    def list_supported_models():
        return list(MODEL_CLASS.keys())

    def fit(
        self,
        train_dataset,
        num_gpus=None,
        gpu_ids=None,
        batch_size=3000,
        local_rank=-1,
        max_steps=5e5,
        warmup_steps=1e5,
        learning_rate=2e-3,
        optimization_method="adam",
        max_grad_norm=0,
        beta1=0.9,
        beta2=0.999,
        decay_method="noam",
        gradient_accumulation_steps=2,
        report_every=50,
        verbose=True,
        seed=None,
        save_every=-1,
        world_size=1,
        **kwargs,
    ):
        """
        Fine-tune pre-trained transofmer models for extractive summarization.

        Args:
            train_dataset (ExtSumProcessedIterableDataset): Training dataset.
            num_gpus (int, optional): The number of GPUs to use.
                If None, all available GPUs will be used. If set to 0 or GPUs are not
                available, CPU device will be used. Defaults to None.
            gpu_ids (list): List of GPU IDs to be used.
                If set to None, the first num_gpus GPUs will be used.
                Defaults to None.
            batch_size (int, optional): Maximum number of tokens in each batch.
            local_rank (int, optional): Local_rank for distributed training on GPUs.
                Defaults to -1, which means non-distributed training.
            max_steps (int, optional): Maximum number of training steps.
                Defaults to 5e5.
            warmup_steps (int, optional): Number of steps taken to increase learning
                rate from 0 to `learning_rate`. Defaults to 1e5.
            learning_rate (float, optional):  Learning rate of the AdamW optimizer.
                Defaults to 5e-5.
            optimization_method (string, optional): Optimization method used in
                fine tuning.
            max_grad_norm (float, optional): Maximum gradient norm for gradient
                clipping.
                Defaults to 0.
            gradient_accumulation_steps (int, optional): Number of batches to accumulate
                gradients on between each model parameter update. Defaults to 1.
            decay_method (string, optional): learning rate decrease method.
                Defaulta to 'noam'.
            report_every (int, optional): The interval by steps to print out the
                trainint log.
                Defaults to 50.
            beta1 (float, optional): The exponential decay rate for the first moment
                estimates.
                Defaults to 0.9.
            beta2 (float, optional): The exponential decay rate for the second-moment
                estimates.
                This value should be set close to 1.0 on problems with a sparse
                gradient.
                Defaults to 0.99.
            verbose (bool, optional): Whether to print out the training log.
                Defaults to True.
            seed (int, optional): Random seed used to improve reproducibility.
                Defaults to None.
        """

        # get device
        device, num_gpus = get_device(
            num_gpus=num_gpus, gpu_ids=gpu_ids, local_rank=local_rank
        )
        # move model
        self.model = move_model_to_device(model=self.model, device=device)

        # init optimizer
        optimizer = model_builder.build_optim(
            optimization_method,
            learning_rate,
            max_grad_norm,
            beta1,
            beta2,
            decay_method,
            warmup_steps,
            self.model,
            None,
        )

        self.model = parallelize_model(
            model=self.model,
            device=device,
            num_gpus=num_gpus,
            gpu_ids=gpu_ids,
            local_rank=local_rank,
        )

        # batch_size is the number of tokens in a batch
        train_dataloader = get_dataloader(
            train_dataset.get_stream(),
            is_labeled=True,
            batch_size=batch_size,
            world_size=world_size,
            rank=local_rank,
        )

        # compute the max number of training steps
        max_steps = compute_training_steps(
            train_dataloader,
            max_steps=max_steps,
            gradient_accumulation_steps=gradient_accumulation_steps,
        )

        super().fine_tune(
            train_dataloader=train_dataloader,
            get_inputs=ExtSumProcessor.get_inputs,
            device=device,
            num_gpus=num_gpus,
            max_steps=max_steps,
            max_grad_norm=max_grad_norm,
            gradient_accumulation_steps=gradient_accumulation_steps,
            optimizer=optimizer,
            scheduler=None,
            verbose=verbose,
            seed=seed,
            report_every=report_every,
            clip_grad_norm=False,
            save_every=save_every,
        )

    def predict(
        self,
        test_dataset,
        num_gpus=1,
        gpu_ids=None,
        batch_size=16,
        sentence_separator="<q>",
        top_n=3,
        block_trigram=True,
        cal_lead=False,
        verbose=True,
    ):
        """
        Predict the summarization for the input data iterator.

        Args:
            test_dataset (Dataset): Dataset for which the summary to be predicted
            num_gpus (int, optional): The number of GPUs used in prediction.
                Defaults to 1.
            gpu_ids (list): List of GPU IDs to be used.
                If set to None, the first num_gpus GPUs will be used.
                Defaults to None.
            batch_size (int, optional): The number of test examples in each batch.
                Defaults to 16.
            sentence_separator (str, optional): String to be inserted between
                sentences in the prediction. Defaults to '<q>'.
            top_n (int, optional): The number of sentences that should be selected
                from the paragraph as summary. Defaults to 3.
            block_trigram (bool, optional): voolean value which specifies whether
                the summary should include any sentence that has the same trigram
                as the already selected sentences. Defaults to True.
            cal_lead (bool, optional): Boolean value which specifies whether the
                prediction uses the first few sentences as summary. Defaults to False.
            verbose (bool, optional): Whether to print out the training log.
                Defaults to True.

        Returns:
            List of strings which are the summaries

        """

        def collate_fn(dict_list):
            # tuple_batch =  [list(col) for col in zip(*[d.values() for d in dict_list]
            if dict_list is None or len(dict_list) <= 0:
                return None
            tuple_batch = [list(d.values()) for d in dict_list]
            # generate mask and mask_cls, and only select tensors for the model input
            # the labels was never used in prediction, set is_labeled as False
            batch = Batch(tuple_batch, is_labeled=False)
            return {
                "src": batch.src,
                "segs": batch.segs,
                "clss": batch.clss,
                "mask": batch.mask,
                "mask_cls": batch.mask_cls,
            }

        test_sampler = SequentialSampler(test_dataset)
        test_dataloader = DataLoader(
            test_dataset,
            sampler=test_sampler,
            batch_size=batch_size,
            collate_fn=collate_fn,
        )
        sent_scores = self.predict_scores(
            test_dataloader, num_gpus=num_gpus, gpu_ids=gpu_ids
        )
        sent_scores_list = list(sent_scores)
        scores_list = []
        for i in sent_scores_list:
            scores_list.extend(i)
        prediction = []
        for i in range(len(test_dataset)):
            temp_pred = get_pred(
                test_dataset[i],
                scores_list[i],
                cal_lead=cal_lead,
                sentence_separator=sentence_separator,
                block_trigram=block_trigram,
                top_n=top_n,
            )
            prediction.extend(temp_pred)
        return prediction

    def predict_scores(self, test_dataloader, num_gpus=1, gpu_ids=None, verbose=True):
        """
        Scores a dataset using a fine-tuned model and a given dataloader.

        Args:
            test_dataloader (Dataloader): Dataloader for scoring the data.
            num_gpus (int, optional): The number of GPUs to use.
                If None, all available GPUs will be used.
                If set to 0 or GPUs are not available, CPU device will be used.
                Defaults to None.
            gpu_ids (list): List of GPU IDs to be used.
                If set to None, the first num_gpus GPUs will be used.
                Defaults to None.
            verbose (bool, optional): Whether to print out the training log.
                Defaults to True.

        Returns
            1darray: numpy array of predicted sentence scores.
        """

        preds = list(
            super().predict(
                eval_dataloader=test_dataloader,
                get_inputs=ExtSumProcessor.get_inputs,
                num_gpus=num_gpus,
                gpu_ids=gpu_ids,
                verbose=verbose,
            )
        )
        return preds

    def save_model(self, full_name=None):
        """
        save the trained model.

        Args:
            full_name (str, optional): File name to save the model's `state_dict()`.
                If it's None, the model is going to be saved under "fine_tuned"
                folder of the cached directory of the object. Defaults to None.
        """
        model_to_save = (
            self.model.module if hasattr(self.model, "module") else self.model
        )  # Take care of distributed/parallel training

        if full_name is None:
            output_model_dir = os.path.join(self.cache_dir, "fine_tuned")
            os.makedirs(self.cache_dir, exist_ok=True)
            os.makedirs(output_model_dir, exist_ok=True)
            full_name = os.path.join(output_model_dir, self.model_name)

        logger.info("Saving model checkpoint to %s", full_name)
        try:
            print("saving through pytorch")
            torch.save(model_to_save.state_dict(), full_name)
        except OSError:
            try:
                print("saving as pickle")
                pickle.dump(model_to_save.state_dict(), open(full_name, "wb"))
            except Exception:
                raise
        except Exception:
            raise
