# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# This script reuses some code from https://github.com/nlpyang/BertSum

import functools
import itertools
import logging
import os
import pickle
from multiprocessing import Pool, cpu_count

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer, BertModel, DistilBertModel

from utils_nlp.common.pytorch_utils import (
    compute_training_steps,
    get_device,
    move_model_to_device,
    parallelize_model,
)
from utils_nlp.dataset.sentence_selection import combination_selection, greedy_selection
from utils_nlp.models.transformers.abstractive_summarization_bertsum import (
    fit_to_block_size,
)

from utils_nlp.models.transformers.bertsum import model_builder
from utils_nlp.models.transformers.bertsum.data_loader import (
    Batch,
    ChunkDataLoader,
    IterableDistributedSampler,
)
from utils_nlp.models.transformers.bertsum.dataset import (
    ExtSumProcessedDataset,
    ExtSumProcessedIterableDataset,
)
from utils_nlp.models.transformers.bertsum.model_builder import BertSumExt
from utils_nlp.models.transformers.common import Transformer

MODEL_CLASS = {
    "bert-base-uncased": BertModel,
    "distilbert-base-uncased": DistilBertModel,
}

logger = logging.getLogger(__name__)


class Bunch(object):
    """ Class which convert a dictionary to an object """

    def __init__(self, adict):
        self.__dict__.update(adict)


def get_dataloader(
    data_iter,
    shuffle=True,
    is_labeled=False,
    batch_size=3000,
    world_size=1,
    rank=0,
    local_rank=-1,
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
    sampler = IterableDistributedSampler(world_size, rank, local_rank)
    return ChunkDataLoader(
        data_iter, batch_size, shuffle=shuffle, is_labeled=is_labeled, sampler=sampler
    )


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
    _pred = []
    final_selections = []
    for j in selected_ids[: len(example["src_txt"])]:
        if j >= len(example["src_txt"]):
            continue
        candidate = example["src_txt"][j].strip()
        if block_trigram:
            if not _block_tri(candidate, _pred):
                _pred.append(candidate)
                final_selections.append(j)
        else:
            _pred.append(candidate)
            final_selections.append(j)

        # only select the top n
        if len(_pred) == top_n:
            break

    sorted_selections = sorted(final_selections)
    _pred = []
    for i in sorted_selections:
        _pred.append(example["src_txt"][i].strip())
    _pred = sentence_separator.join(_pred)
    pred.append(_pred.strip())
    return pred


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

    def splits(self, root, train_iterable=False):
        """Get the train and test dataset from the folder

        Args:
            root (str): Directory where the data can be loaded.

        Returns:
            Tuple of ExtSumProcessedIterableDataset as train dataset
            and ExtSumProcessedDataset as test dataset.
        """
        train_files, test_files = self._get_files(root)
        if train_iterable:
            return (
                ExtSumProcessedIterableDataset(train_files, is_shuffle=True),
                ExtSumProcessedDataset(test_files, is_shuffle=False),
            )
        else:
            return (
                ExtSumProcessedDataset(train_files, is_shuffle=True),
                ExtSumProcessedDataset(test_files, is_shuffle=False),
            )


def preprocess_single_add_oracleids(input_data, oracle_mode="greedy", selections=3):
    """ Preprocess single data point to generate oracle summaries and
        sentence tokenization of the source text.

        Args:
            input_data (dict): An item from `SummarizationDataset`
            oracle_mode (str, optional): Sentence selection method.
                Defaults to "greedy".
            selections (int, optional): The number of sentence used as summary.
                Defaults to 3.
        Returns:
            Dictionary of fields "src", "src_txt", "tgt", "tgt_txt" and "oracle_ids"
    """

    oracle_ids = None
    if "tgt" in input_data:
        if oracle_mode == "greedy":
            oracle_ids = greedy_selection(
                input_data["src"], input_data["tgt"], selections
            )
        elif oracle_mode == "combination":
            oracle_ids = combination_selection(
                input_data["src"], input_data["tgt"], selections
            )
        input_data["oracle_ids"] = oracle_ids
    # input_data["src_txt"] = tokenize.sent_tokenize(input_data["src_txt"])
    return input_data


def parallel_preprocess(input_data, preprocess, num_pool=-1):
    """
    Process data in parallel using multiple GPUs.

    Args:
        input_data (list): List if input strings to process.
        preprocess_pipeline (list): List of functions to apply on the input data.
        word_tokenize (func, optional): A tokenization function used to tokenize
            the results from preprocess_pipeline.
        num_pool (int, optional): Number of CPUs to use. Defaults to -1 and all
            available CPUs are used.

    Returns:
        list: list of processed text strings.

    """
    if num_pool == -1:
        num_pool = cpu_count()

    num_pool = min(num_pool, len(input_data))

    p = Pool(num_pool)

    results = p.map(
        preprocess, input_data, chunksize=min(1, int(len(input_data) / num_pool))
    )
    p.close()
    p.join()

    return results


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
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            do_lower_case=to_lower,
            cache_dir=cache_dir,
            output_loading_info=False,
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
        return list(MODEL_CLASS)

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
            model_name (bool): Model name used to format the inputs.
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
                batch = batch.to(device)
                return {
                    "x": batch.src,
                    "segs": batch.segs,
                    "clss": batch.clss,
                    "mask": batch.mask,
                    "mask_cls": batch.mask_cls,
                    # "labels": batch.labels,
                }
                """
                return {
                    "x": batch.src.to(device),
                    "segs": batch.segs.to(device),
                    "clss": batch.clss.to(device),
                    "mask": batch.mask.to(device),
                    "mask_cls": batch.mask_cls.to(device),
                }
                """
        else:
            raise ValueError("Model not supported: {}".format(model_name))

    def preprocess(self, input_data_list, oracle_mode="greedy", selections=3):
        """ Preprocess multiple data points.

           Args:
              input_data_list (SummarizationDataset): The dataset to be preprocessed.
              oracle_mode (str, optional): Sentence selection method.
                Defaults to "greedy".
              selections (int, optional): The number of sentence used as summary.
                Defaults to 3.

            Returns:
                Iterator of dictory objects containing input ids, segment ids,
                sentence class ids, labels, source text and target text.
                If targets is None, the label and target text are None.
        """
        preprocess = functools.partial(
            preprocess_single_add_oracleids, oracle_mode="greedy", selections=3
        )
        return parallel_preprocess(input_data_list, preprocess)

    def collate(self, data, block_size, device, train_mode=True):
        """ Collcate function for pytorch data loaders.
            Args:
                data (list): A list of samples from SummarizationDataset.
                block_size (int): maximum input length for the model.
                train_mode (bool): whether the collate function is used for training
                    or not. Defaults to True.

            Returns:
                `Batch` object: a data minibatch as the input of a model.

        """

        if len(data) == 0:
            return None
        else:
            if train_mode is True and "tgt" in data[0] and "oracle_ids" in data[0]:
                encoded_text = [self.encode_single(d, block_size) for d in data]
                batch = Batch(list(filter(None, encoded_text)), True)
            else:
                encoded_text = [
                    self.encode_single(d, block_size, train_mode) for d in data
                ]
                # src, labels, segs, clss, src_txt, tgt_txt =  zip(*encoded_text)
                # new_data = [list(i) for i in list(zip(*encoded_text))]
                # batch =  Batch(new_data)
                filtered_list = list(filter(None, encoded_text))
                # if len(filtered_list) != len(data):
                #    raise ValueError("no test data shouldn't be skipped")
                batch = Batch(filtered_list)
            return batch.to(device)

    def encode_single(self, d, block_size, train_mode=True):
        """ Enocde a single sample.
            Args:
                d (dict): s data sample from SummarizationDataset.
                block_size (int): maximum input length for the model.

            Returns:
                Tuple of encoded data.

        """

        src = d["src"]

        if len(src) == 0:
            raise ValueError("source doesn't have any sentences")

        original_src_txt = [" ".join(s) for s in src]
        # no filtering for prediction
        idxs = [i for i, s in enumerate(src)]
        src = [src[i] for i in idxs]

        tgt_txt = None
        labels = None
        if (
            train_mode and "oracle_ids" in d and "tgt" in d and "tgt_txt" in d
        ):  # is not None and tgt is not None:
            labels = [0] * len(src)
            for l in d["oracle_ids"]:
                labels[l] = 1

            # source filtering for only training
            idxs = [i for i, s in enumerate(src) if (len(s) > self.min_src_ntokens)]
            src = [src[i][: self.max_src_ntokens] for i in idxs]
            src = src[: self.max_nsents]
            labels = [labels[i] for i in idxs]
            labels = labels[: self.max_nsents]

            if len(src) < self.min_nsents:
                return None
            if len(labels) == 0:
                return None
            tgt_txt = "".join([" ".join(tt) for tt in d["tgt"]])

        src_txt = [" ".join(sent) for sent in src]
        text = " [SEP] [CLS] ".join(src_txt)
        src_subtokens = self.tokenizer.tokenize(text)
        # src_subtokens = src_subtokens[:510]
        src_subtokens = (
            ["[CLS]"]
            + fit_to_block_size(
                src_subtokens, block_size - 2, self.tokenizer.pad_token_id
            )
            + ["[SEP]"]
        )
        src_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(src_subtokens)
        _segs = [-1] + [i for i, t in enumerate(src_subtoken_idxs) if t == self.sep_vid]
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
        src_txt = [original_src_txt[i] for i in idxs]
        return src_subtoken_idxs, labels, segments_ids, cls_ids, src_txt, tgt_txt


class ExtractiveSummarizer(Transformer):
    """class which performs extractive summarization fine tuning and prediction """

    def __init__(
        self,
        processor,
        model_name="distilbert-base-uncased",
        encoder="transformer",
        max_pos_length=512,
        cache_dir=".",
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

        model = MODEL_CLASS[model_name].from_pretrained(
            model_name, cache_dir=cache_dir, num_labels=0, output_loading_info=False
        )
        super().__init__(model_name=model_name, model=model, cache_dir=cache_dir)

        if model_name not in self.list_supported_models():
            raise ValueError(
                "Model name {} is not supported by ExtractiveSummarizer. "
                "Call 'ExtractiveSummarizer.list_supported_models()' to get all  "
                "supported model names.".format(model_name)
            )
        self.processor = processor
        self.max_pos_length = max_pos_length
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
        self.model = BertSumExt(
            encoder, args, self.model_class, model_name, max_pos_length, None, cache_dir
        )

    @staticmethod
    def list_supported_models():
        return list(MODEL_CLASS)

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
        gradient_accumulation_steps=1,
        report_every=50,
        verbose=True,
        seed=None,
        save_every=-1,
        world_size=1,
        rank=0,
        use_preprocessed_data=False,
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
            rank (int, optional): Global rank of the current GPU in distributed
                training. It's calculated with the rank of the current node in
                the cluster/world and the `local_rank` of the device in the current
                node. See an example in :file: `examples/text_summarization/
                extractive_summarization_cnndm_distributed_train.py`.
                Defaults to 0.
        """

        # get device
        device, num_gpus = get_device(
            num_gpus=num_gpus, gpu_ids=gpu_ids, local_rank=local_rank
        )
        # move model
        self.model = move_model_to_device(model=self.model, device=device)

        # init optimizer
        optimizer = model_builder.build_optim(
            self.model,
            optimization_method,
            learning_rate,
            max_grad_norm,
            beta1,
            beta2,
            decay_method,
            warmup_steps,
        )
        self.model = parallelize_model(
            model=self.model,
            device=device,
            num_gpus=num_gpus,
            gpu_ids=gpu_ids,
            local_rank=local_rank,
        )

        # batch_size is the number of tokens in a batch
        if use_preprocessed_data:
            train_dataloader = get_dataloader(
                train_dataset.get_stream(),
                is_labeled=True,
                batch_size=batch_size,
                world_size=world_size,
                rank=rank,
                local_rank=local_rank,
            )
        else:
            if local_rank == -1:
                sampler = RandomSampler(train_dataset)
            else:
                sampler = DistributedSampler(
                    train_dataset, num_replicas=world_size, rank=rank
                )

            def collate_fn(data):
                return self.processor.collate(
                    data, block_size=self.max_pos_length, device=device
                )

            train_dataloader = DataLoader(
                train_dataset,
                sampler=sampler,
                batch_size=batch_size,
                collate_fn=collate_fn,
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
        num_gpus=None,
        gpu_ids=None,
        batch_size=16,
        sentence_separator="<q>",
        top_n=3,
        block_trigram=True,
        cal_lead=False,
        verbose=True,
        local_rank=-1,
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

        device, num_gpus = get_device(
            num_gpus=num_gpus, gpu_ids=gpu_ids, local_rank=local_rank
        )

        def collate_processed_data(dict_list):
            # tuple_batch =  [list(col) for col in zip(*[d.values() for d in dict_list]
            if dict_list is None or len(dict_list) <= 0:
                return None
            tuple_batch = [list(d.values()) for d in dict_list]
            # generate mask and mask_cls, and only select tensors for the model input
            # the labels was never used in prediction, set is_labeled as False
            batch = Batch(tuple_batch, is_labeled=False)
            return batch

        def collate(data):
            return self.processor.collate(
                data, block_size=self.max_pos_length, train_mode=False, device=device
            )

        if len(test_dataset) == 0:
            return None
        if "segs" in test_dataset[0]:
            collate_fn = collate_processed_data
        else:
            collate_fn = collate

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

        # release GPU memories
        self.model.cpu()
        torch.cuda.empty_cache()

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
