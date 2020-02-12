# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# This script reuses some code from https://github.com/nlpyang/Presumm
# HuggingFace's
# Add to noticefile

from collections import namedtuple
import itertools
import logging
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset, SequentialSampler

# from torch.utils.data.distributed import DistributedSampler
from transformers import BertModel, DistilBertModel

from bertsum.models import data_loader, model_builder
from bertsum.models.data_loader import Batch
from bertsum.models.model_builder import Summarizer
from utils_nlp.common.pytorch_utils import compute_training_steps, get_device
from utils_nlp.dataset.sentence_selection import combination_selection, greedy_selection
from utils_nlp.models.transformers.common import TOKENIZER_CLASS, Transformer

from .extractive_summarization import Bunch

from torch.utils.data import SequentialSampler, RandomSampler, DataLoader

MODEL_CLASS = {"bert-base-uncased": BertModel, "distilbert-base-uncased": DistilBertModel}

logger = logging.getLogger(__name__)

import sys

# sys.path.insert(0, "/dadendev/PreSumm2/PreSumm/src")
# sys.path.insert(0, "/dadendev/PreSumm2/PreSumm/src/models")
from utils_nlp.models.transformers.bertabs import model_builder
from utils_nlp.models.transformers.bertabs.model_builder import AbsSummarizer
from utils_nlp.models.transformers.bertabs.loss import abs_loss
from utils_nlp.models.transformers.bertabs.predictor import build_predictor

from utils_nlp.dataset.cnndm import CNNDMBertSumProcessedData, CNNDMSummarizationDataset
from utils_nlp.models.transformers.datasets import SummarizationNonIterableDataset
from utils_nlp.eval.evaluate_summarization import get_rouge

from tempfile import TemporaryDirectory


def shorten_dataset(dataset, top_n=-1):
    if top_n == -1:
        return dataset
    return SummarizationNonIterableDataset(dataset.source[0:top_n], dataset.target[0:top_n])


def validation(saved_model_path):
    TOP_N = 10
    processor = AbsSumProcessor(cache_dir="/dadendev/nlp/examples/text_summarization/")
    test_sum_dataset = torch.load(
        "/dadendev/nlp/examples/text_summarization/test_abssum_dataset_full.pt"
    )
    summarizer = AbsSum(
        processor,
        checkpoint=torch.load(saved_model_path),
        cache_dir="/dadendev/nlp/examples/text_summarization/",
    )

    src = test_sum_dataset.source[0:TOP_N]
    reference_summaries = ["".join(t).rstrip("\n") for t in test_sum_dataset.target[0:TOP_N]]
    generated_summaries = summarizer.predict(
        shorten_dataset(test_sum_dataset, top_n=TOP_N), batch_size=8
    )
    assert len(generated_summaries) == len(reference_summaries)
    for i in generated_summaries[0:1]:
        print(i)
        print("\n")
        print("###################")

    for i in reference_summaries[0:1]:
        print(i)
        print("\n")

    RESULT_DIR = TemporaryDirectory().name
    rouge_score = get_rouge(generated_summaries, reference_summaries, RESULT_DIR)
    return "rouge score: {}".format(rouge_score)


def fit_to_block_size(sequence, block_size, pad_token_id):
    """ Adapt the source and target sequences' lengths to the block size.
    If the sequence is shorter we append padding token to the right of the sequence.
    """
    if len(sequence) > block_size:
        return sequence[:block_size]
    else:
        sequence.extend([pad_token_id] * (block_size - len(sequence)))
        return sequence


def build_mask(sequence, pad_token_id):
    """ Builds the mask. The attention mechanism will only attend to positions
    with value 1. """
    mask = torch.ones_like(sequence)
    idx_pad_tokens = sequence == pad_token_id
    mask[idx_pad_tokens] = 0
    return mask


def compute_token_type_ids(batch, separator_token_id):
    """ Segment embeddings as described in [1]
    The values {0,1} were found in the repository [2].
    Attributes:
        batch: torch.Tensor, size [batch_size, block_size]
            Batch of input.
        separator_token_id: int
            The value of the token that separates the segments.
    [1] Liu, Yang, and Mirella Lapata. "Text summarization with pretrained encoders."
        arXiv preprint arXiv:1908.08345 (2019).
    [2] https://github.com/nlpyang/PreSumm (/src/prepro/data_builder.py, commit fac1217)
    """
    batch_embeddings = []
    for sequence in batch:
        sentence_num = -1
        embeddings = []
        for s in sequence:
            if s == separator_token_id:
                sentence_num += 1
            embeddings.append(sentence_num % 2)
        batch_embeddings.append(embeddings)
    return torch.tensor(batch_embeddings)


class AbsSumProcessor:
    """Class for preprocessing extractive summarization data."""

    def __init__(
        self,
        model_name="bert-base-uncased",
        to_lower=False,
        cache_dir=".",
        max_len=512,
        max_target_len=140,
    ):
        """ Initialize the preprocessor.

        Args:
            model_name (str, optional): Transformer model name used in preprocessing.
                check MODEL_CLASS for supported models. Defaults to "bert-base-cased".
            to_lower (bool, optional): Whether to convert all letters to lower case during
                tokenization. This is determined by if a cased model is used.
                Defaults to False, which corresponds to a cased model.
            cache_dir (str, optional): Directory to cache the tokenizer. Defaults to ".".
            max_src_ntokens (int, optional): Max number of tokens that be used
                as input. Defaults to 512.
        
        """
        self.model_name = model_name
        self.tokenizer = TOKENIZER_CLASS[self.model_name].from_pretrained(
            self.model_name, do_lower_case=to_lower, cache_dir=cache_dir
        )

        self.symbols = {
            "BOS": self.tokenizer.vocab["[unused0]"],
            "EOS": self.tokenizer.vocab["[unused1]"],
            "PAD": self.tokenizer.vocab["[PAD]"],
            "EOQ": self.tokenizer.vocab["[unused2]"],
        }

        self.sep_token = "[SEP]"
        self.cls_token = "[CLS]"
        self.pad_token = "[PAD]"
        self.tgt_bos = "[unused0]"
        self.tgt_eos = "[unused1]"

        self.sep_vid = self.tokenizer.vocab[self.sep_token]
        self.cls_vid = self.tokenizer.vocab[self.cls_token]
        self.pad_vid = self.tokenizer.vocab[self.pad_token]

        self.max_len = max_len
        self.max_target_len = max_target_len

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
                "Call 'ExtSumProcessor.list_supported_models()' to get all supported model "
                "names.".format(value)
            )

        self._model_name = value

    @staticmethod
    def get_inputs(batch, device, model_name, train_mode=True):
        """
        Creates an input dictionary given a model name.

        Args:
            batch (object): A Batch containing input ids, segment ids, sentence class ids,
                masks for the input ids, masks for  sentence class ids and source text.
                If train_model is True, it also contains the labels and target text.
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
                # labels must be the last

                return {
                    "src": batch.src,
                    "segs": batch.segs,
                    "mask_src": batch.mask_src,
                    "tgt": batch.tgt,
                    # "tgt_num_tokens": batch.tgt_num_tokens
                }
            else:
                return {
                    "src": batch.src,
                    "segs": batch.segs,
                    "mask_src": batch.mask_src,
                }
        else:
            raise ValueError("Model not supported: {}".format(model_name))

    def collate(self, data, block_size, device, train_mode=True):
        """ Collate formats the data passed to the data loader.
        In particular we tokenize the data batch after batch to avoid keeping them
        all in memory. We output the data as a namedtuple to fit the original BertAbs's
        API.
        """
        data = [x for x in data if not len(x[1]) == 0]  # remove empty_files
        stories = [" ".join(story) for story, _ in data]
        summaries = [" ".join(summary) for _, summary in data]

        encoded_text = [self.preprocess(story, summary) for story, summary in data]
        # print(encoded_text[0])

        # """"""
        encoded_stories = torch.tensor(
            [
                fit_to_block_size(story, block_size, self.tokenizer.pad_token_id)
                for story, _ in encoded_text
            ]
        )
        encoder_token_type_ids = compute_token_type_ids(
            encoded_stories, self.tokenizer.cls_token_id
        )
        encoder_mask = build_mask(encoded_stories, self.tokenizer.pad_token_id)
        # """

        if train_mode:
            encoded_summaries = torch.tensor(
                [
                    fit_to_block_size(summary, block_size, self.tokenizer.pad_token_id)
                    for _, summary in encoded_text
                ]
            )
            summary_num_tokens = [
                encoded_summary.ne(self.tokenizer.pad_token_id).sum()
                for encoded_summary in encoded_summaries
            ]
            # print(summary_num_tokens)

            Batch = namedtuple(
                "Batch",
                [
                    "batch_size",
                    "src",
                    "segs",
                    "mask_src",
                    "tgt",
                    "tgt_num_tokens",
                    "src_str",
                    "tgt_str",
                ],
            )
            batch = Batch(
                # document_names=None,
                batch_size=len(encoded_stories),
                src=encoded_stories.to(device),
                segs=encoder_token_type_ids.to(device),
                mask_src=encoder_mask.to(device),
                tgt_num_tokens=torch.stack(summary_num_tokens).to(device),
                tgt=encoded_summaries.to(device),
                src_str=stories,
                tgt_str=summaries,
            )
        else:
            Batch = namedtuple("Batch", ["batch_size", "src", "segs", "mask_src"])
            batch = Batch(
                # document_names=None,
                batch_size=len(encoded_stories),
                src=encoded_stories.to(device),
                segs=encoder_token_type_ids.to(device),
                mask_src=encoder_mask.to(device),
            )

        return batch

    def preprocess(self, story_lines, summary_lines=None):
        """preprocess multiple data points

           Args:
              sources (list of list of strings): List of word tokenized sentences.
              targets (list of list of strings, optional): List of word tokenized sentences.
                  Defaults to None, which means it doesn't include summary and is
                  not training data.

            Returns:
                Iterator of dictory objects containing input ids, segment ids, sentence class ids,
                labels, source text and target text. If targets is None, the label and target text
                are None.
        """
        # story_lines_token_ids = [self.tokenizer.encode(line, max_length=self.max_len) for line in story_lines]
        story_lines_token_ids = []
        for line in story_lines:
            try:
                if len(line) <= 0:
                    continue
                story_lines_token_ids.append(self.tokenizer.encode(line, max_length=self.max_len))
            except:
                print(line)
                raise
        story_token_ids = [token for sentence in story_lines_token_ids for token in sentence]
        if summary_lines:
            summary_lines_token_ids = []
            for line in summary_lines:
                try:
                    if len(line) <= 0:
                        continue
                    summary_lines_token_ids.append(
                        self.tokenizer.encode(line, max_length=self.max_target_len)
                    )
                except:
                    print(line)
                    raise
            summary_token_ids = [
                token for sentence in summary_lines_token_ids for token in sentence
            ]
            return story_token_ids, summary_token_ids
        else:
            return story_token_ids


class AbsSum(Transformer):
    """class which performs extractive summarization fine tuning and prediction """

    def __init__(
        self,
        processor,
        model_name="bert-base-uncased",
        encoder="encoder",
        finetune_bert=True,
        label_smoothing=0.1,
        cache_dir=".",
        checkpoint=None,
    ):
        """Initialize a ExtractiveSummarizer.

        Args:
            model_name (str, optional): Transformer model name used in preprocessing.
                check MODEL_CLASS for supported models. Defaults to "distilbert-base-uncased".
            encoder (str, optional): Encoder algorithm used by summarization layer.
                There are four options:
                    - baseline: it used a smaller transformer model to replace the bert model
                      and with transformer summarization layer.
                    - classifier: it uses pretrained BERT and fine-tune BERT with simple logistic
                      classification summarization layer.
                    - transformer: it uses pretrained BERT and fine-tune BERT with transformer
                      summarization layer.
                    - RNN: it uses pretrained BERT and fine-tune BERT with LSTM summarization layer.
                Defaults to "transformer".
            cache_dir (str, optional): Directory to cache the tokenizer. Defaults to ".".
        """

        super().__init__(
            model_class=MODEL_CLASS, model_name=model_name, num_labels=0, cache_dir=cache_dir
        )
        if model_name not in self.list_supported_models():
            raise ValueError(
                "Model name {} is not supported by ExtractiveSummarizer. "
                "Call 'ExtractiveSummarizer.list_supported_models()' to get all supported model "
                "names.".format(value)
            )

        self.model_class = MODEL_CLASS[model_name]
        self.checkpoint = checkpoint
        from utils_nlp.common.pytorch_utils import get_device

        device, num_gpus = get_device(num_gpus=4, local_rank=-1)
        self.model = AbsSummarizer(
            device,
            temp_dir=cache_dir,
            encoder=encoder,
            finetune_bert=finetune_bert,
            checkpoint=checkpoint,
        )
        self.device = device
        self.processor = processor
        self.optim_bert = None
        self.optim_dec = None

        self.train_loss = abs_loss(
            self.model.generator,
            self.processor.symbols,
            self.model.vocab_size,
            device,
            train=True,
            label_smoothing=label_smoothing,
        )

    @staticmethod
    def list_supported_models():
        return list(MODEL_CLASS.keys())

    def fit(
        self,
        train_dataset,
        num_gpus=None,
        gpu_ids=None,
        batch_size=140,
        local_rank=-1,
        max_steps=5e5,
        warmup_steps_bert=8000,
        warmup_steps_dec=8000,
        learning_rate_bert=0.002,
        learning_rate_dec=0.2,
        optimization_method="adam",
        max_grad_norm=0,
        beta1=0.9,
        beta2=0.999,
        decay_method="noam",
        gradient_accumulation_steps=2,
        report_every=10,
        save_every=100,
        verbose=True,
        seed=None,
        fp16=False,
        validation_function=None,
        **kwargs,
    ):
        """
        Fine-tune pre-trained transofmer models for extractive summarization.

        Args:
            train_dataset (ExtSumProcessedIterableDataset): Training dataset.
            num_gpus (int, optional): The number of GPUs to use. If None, all available GPUs will
                be used. If set to 0 or GPUs are not available, CPU device will
                be used. Defaults to None.
            gpu_ids (list): List of GPU IDs to be used.
                If set to None, the first num_gpus GPUs will be used.
                Defaults to None.
            batch_size (int, optional): Maximum number of tokens in each batch.
            local_rank (int, optional): Local_rank for distributed training on GPUs. Defaults to
                -1, which means non-distributed training.
            max_steps (int, optional): Maximum number of training steps. Defaults to 5e5.
            warmup_steps (int, optional): Number of steps taken to increase learning rate from 0
                to `learning_rate`. Defaults to 1e5.
            learning_rate (float, optional):  Learning rate of the AdamW optimizer. Defaults to
                5e-5.
            optimization_method (string, optional): Optimization method used in fine tuning.
            max_grad_norm (float, optional): Maximum gradient norm for gradient clipping.
                Defaults to 0.
            gradient_accumulation_steps (int, optional): Number of batches to accumulate
                gradients on between each model parameter update. Defaults to 1.
            decay_method (string, optional): learning rate decrease method. Default to 'noam'.
            report_every (int, optional): The interval by steps to print out the trainint log.
                Defaults to 50.
            beta1 (float, optional): The exponential decay rate for the first moment estimates.
                Defaults to 0.9.
            beta2 (float, optional): The exponential decay rate for the second-moment estimates.
                This value should be set close to 1.0 on problems with a sparse gradient.
                Defaults to 0.99.
            verbose (bool, optional): Whether to print out the training log. Defaults to True.
            seed (int, optional): Random seed used to improve reproducibility. Defaults to None.
        """

        # init optimizer

        self.optim_bert = model_builder.build_optim_bert(
            self.model,
            visible_gpus="0,1,2,3",
            lr_bert=learning_rate_bert,
            warmup_steps_bert=warmup_steps_bert,
            checkpoint=self.checkpoint,
        )
        self.optim_dec = model_builder.build_optim_dec(
            self.model,
            visible_gpus="0,1,2,3",
            lr_dec=learning_rate_dec,
            warmup_steps_dec=warmup_steps_dec,
            checkpoint=self.checkpoint,
        )
        optim = [self.optim_bert, self.optim_dec]

        def build_data_iterator(collate, dataset, batch_size=16, device="cuda"):

            sampler = RandomSampler(dataset)

            def collate_fn(data):
                return collate(data, block_size=512, device=device)

            iterator = DataLoader(
                dataset, sampler=sampler, batch_size=batch_size, collate_fn=collate_fn,
            )

            return iterator

        # batch_size is the number of tokens in a batch
        # train_dataloader = get_dataloader(train_dataset.get_stream(), is_labeled=True, batch_size=batch_size)
        train_dataloader = build_data_iterator(
            self.processor.collate, train_dataset, batch_size=batch_size, device=self.device
        )

        # compute the max number of training steps
        max_steps = compute_training_steps(
            train_dataloader,
            max_steps=max_steps,
            gradient_accumulation_steps=gradient_accumulation_steps,
        )

        def loss(inputs, outputs):
            # return  self.train_loss.sharded_compute_loss(inputs, outputs, Bunch({"generator_shard_size":32}), normalization)
            return self.train_loss.monolithic_compute_loss(inputs, outputs)

        super().fine_tune(
            train_dataloader=train_dataloader,
            get_inputs=AbsSumProcessor.get_inputs,
            num_gpus=num_gpus,
            gpu_ids=gpu_ids,
            max_steps=max_steps,
            max_grad_norm=max_grad_norm,
            gradient_accumulation_steps=gradient_accumulation_steps,
            scheduler=None,
            verbose=verbose,
            seed=seed,
            report_every=report_every,
            save_every=save_every,
            clip_grad_norm=False,
            optimizer=optim,
            loss_function=loss,
            fp16=fp16,
            validation_function=validation_function,
        )

    def predict(
        self,
        test_dataset,
        num_gpus=1,
        gpu_ids=None,
        batch_size=16,
        # sentence_separator="<q>",
        alpha=0.6,
        beam_size=5,
        min_length=15,
        max_length=150,
        verbose=True,
    ):
        """
        Predict the summarization for the input data iterator.

        Args:
            test_dataset (Dataset): Dataset for which the summary to be predicted
            num_gpus (int, optional): The number of GPUs used in prediction. Defaults to 1.
            gpu_ids (list): List of GPU IDs to be used.
                If set to None, the first num_gpus GPUs will be used.
                Defaults to None.
            batch_size (int, optional): The number of test examples in each batch. Defaults to 16.
            sentence_separator (str, optional): String to be inserted between sentences in
                the prediction. Defaults to '<q>'.
            top_n (int, optional): The number of sentences that should be selected
                from the paragraph as summary. Defaults to 3.
            block_trigram (bool, optional): voolean value which specifies whether
                the summary should include any sentence that has the same trigram
                as the already selected sentences. Defaults to True.
            cal_lead (bool, optional): Boolean value which specifies whether the
                prediction uses the first few sentences as summary. Defaults to False.
            verbose (bool, optional): Whether to print out the training log. Defaults to True.

        Returns:
            List of strings which are the summaries

        """

        def format_summary(translation):
            """ Transforms the output of the `from_batch` function
            into nicely formatted summaries.
            """
            raw_summary, _, = translation
            summary = (
                raw_summary.replace("[unused0]", "")
                .replace("[unused3]", "")
                .replace("[PAD]", "")
                .replace("[unused1]", "")
                .replace(r" +", " ")
                .replace(" [unused2] ", ". ")
                .replace("[unused2]", "")
                .strip()
            )

            return summary

        test_sampler = SequentialSampler(test_dataset)

        def collate_fn(data):
            return self.processor.collate(data, 512, self.device, train_mode=False)

        test_dataloader = DataLoader(
            test_dataset, sampler=test_sampler, batch_size=batch_size, collate_fn=collate_fn,
        )
        predictor = build_predictor(
            self.processor.tokenizer,
            self.processor.symbols,
            self.model,
            alpha=alpha,
            beam_size=beam_size,
            min_length=min_length,
            max_length=max_length,
        )

        generated_summaries = []
        from tqdm import tqdm

        for batch in tqdm(test_dataloader):
            batch_data = predictor.translate_batch(batch)
            translations = predictor.from_batch(batch_data)
            summaries = [format_summary(t) for t in translations]
            generated_summaries += summaries
        return generated_summaries

    def save_model(self, full_name=None):
        """
        save the trained model.

        Args:
            full_name (str, optional): File name to save the model's `state_dict()`. If it's None,
                the model is going to be saved under "fine_tuned" folder of the cached directory
                of the object. Defaults to None.
        """
        save_obj = {"optims": [self.optim_bert, self.optim_dec]}
        model_to_save = (
            self.model.module if hasattr(self.model, "module") else self.model
        )  # Take care of distributed/parallel training

        if full_name is None:
            output_model_dir = os.path.join(self.cache_dir, "fine_tuned")
            os.makedirs(self.cache_dir, exist_ok=True)
            os.makedirs(output_model_dir, exist_ok=True)
            full_name = os.path.join(output_model_dir, name)

        save_obj["model"] = model_to_save.state_dict()
        logger.info("Saving model checkpoint to %s", full_name)
        try:
            print("saving through pytorch")
            torch.save(save_obj, full_name)
        except OSError:
            try:
                print("saving as pickle")
                pickle.dump(save_obj, open(full_name, "wb"))
            except Exception:
                raise
        except Exception:
            raise
