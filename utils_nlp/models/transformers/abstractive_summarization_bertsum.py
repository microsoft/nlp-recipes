# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# This script reuses some code from https://github.com/nlpyang/Presumm
# This script reuses some code from https://github.com/huggingface/transformers/
# Add to noticefile

import logging
import os
import pickle
from collections import namedtuple

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import AutoTokenizer, BertModel

from utils_nlp.common.pytorch_utils import (
    compute_training_steps,
    get_amp,
    get_device,
    move_model_to_device,
    parallelize_model,
)
from utils_nlp.eval import compute_rouge_python
from utils_nlp.models.transformers.bertsum import model_builder
from utils_nlp.models.transformers.bertsum.model_builder import AbsSummarizer
from utils_nlp.models.transformers.bertsum.predictor import build_predictor
from utils_nlp.models.transformers.common import Transformer

MODEL_CLASS = {"bert-base-uncased": BertModel}

logger = logging.getLogger(__name__)


def fit_to_block_size(sequence, block_size, pad_token_id):
    """ Adapt the source and target sequences' lengths to the block size.
    If the sequence is shorter we append padding token to the right of the sequence.

    Args:
        sequence (list): sequence to be truncated to padded
        block_size (int): length of the output

    Returns:
        sequence (list): padded or shortend list

    """
    if len(sequence) > block_size:
        return sequence[:block_size]
    else:
        sequence.extend([pad_token_id] * (block_size - len(sequence)))
        return sequence


def build_mask(sequence, pad_token_id):
    """ Builds the mask. The attention mechanism will only attend to positions
    with value 1.

    Args:
        sequence (list): sequences for which the mask is built for.
        pad_token_id (long): padding token id for which the mask is 0.

    Returns:
        mask (list): sequences of 1s and 0s.

    """
    mask = torch.ones_like(sequence)
    idx_pad_tokens = sequence == pad_token_id
    mask[idx_pad_tokens] = 0
    return mask


def compute_token_type_ids(batch, separator_token_id):
    """ Segment embeddings as described in [1]
    The values {0,1} were found in the repository [2].

    Args:
        batch (torch.Tensor, size [batch_size, block_size]):
            Batch of input.
        separator_token_id: int
            The value of the token that separates the segments.

    Returns:
        torch.Tensor, size [batch_size, block_size]): segment embeddings.

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


class BertSumAbsProcessor:
    """Class for preprocessing abstractive summarization data for
        BertSumAbs algorithm."""

    def __init__(
        self,
        model_name="bert-base-uncased",
        to_lower=True,
        cache_dir=".",
        max_src_len=640,
        max_tgt_len=140,
    ):
        """ Initialize the preprocessor.

        Args:
            model_name (str, optional): Transformer model name used in preprocessing.
                check MODEL_CLASS for supported models. Defaults to "bert-base-cased".
            to_lower (bool, optional): Whether to convert all letters to lower case
                during tokenization. This is determined by if a cased model is used.
                Defaults to True, which corresponds to a uncased model.
            cache_dir (str, optional): Directory to cache the tokenizer.
                Defaults to ".".
            max_src_len (int, optional): Max number of tokens that be used
                as input. Defaults to 640.
            max_tgt_len (int, optional): Max number of tokens that be used
                as in target. Defaults to 140.

        """
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            do_lower_case=to_lower,
            cache_dir=cache_dir,
            output_loading_info=False,
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
        self.tgt_bos = self.symbols["BOS"]
        self.tgt_eos = self.symbols["EOS"]

        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len

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
                "Model name {} is not supported by BertSumAbsProcessor. "
                "Call 'BertSumAbsProcessor.list_supported_models()' to "
                "get all supported model names.".format(value)
            )

        self._model_name = value

    @staticmethod
    def get_inputs(batch, device, model_name, train_mode=True):
        """
        Creates an input dictionary given a model name.

        Args:
            batch (object): A Batch containing input ids, segment ids,
                masks for the input ids and source text. If train_mode is True, it
                also contains the target ids and the number of tokens
                in the target and target text.
            device (torch.device): A PyTorch device.
            model_name (bool): Model name used to format the inputs.
            train_mode (bool, optional): Training mode flag.
                Defaults to True.

        Returns:
            dict: Dictionary containing input ids, segment ids, sentence class ids,
            masks for the input ids. Target ids and number of tokens in the target are
            only returned when train_mode is True.
        """

        if model_name.split("-")[0] in ["bert"]:
            if train_mode:
                # labels must be the last

                return {
                    "src": batch.src,
                    "segs": batch.segs,
                    "mask_src": batch.mask_src,
                    "tgt": batch.tgt,
                    "tgt_num_tokens": batch.tgt_num_tokens,
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
        all in memory.

        Args:
            data (list of (str, str)): input data to be loaded.
            block_size (long): size of the encoded data to be passed into the data loader
            device (torch.device): A PyTorch device.
            train_mode (bool, optional): Training mode flag.
                Defaults to True.

        Returns:
            namedtuple: a nametuple containing input ids, segment ids,
                masks for the input ids and source text. If train_mode is True, it
                also contains the target ids and the number of tokens
                in the target and target text.
        """
        data = [x for x in data if not len(x["src"]) == 0]  # remove empty_files
        if len(data) == 0:
            return None
        stories = [" ".join(d["src"]) for d in data]
        if train_mode is True and "tgt" in data[0]:
            summaries = [" ".join(d["tgt"]) for d in data]
            encoded_text = [self.preprocess(d["src"], d["tgt"]) for d in data]
        else:
            encoded_text = [self.preprocess(d["src"], None) for d in data]

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

        if train_mode and "tgt" in data[0]:
            encoded_summaries = torch.tensor(
                [
                    [self.tgt_bos]
                    + fit_to_block_size(
                        summary, block_size - 2, self.tokenizer.pad_token_id
                    )
                    + [self.tgt_eos]
                    for _, summary in encoded_text
                ]
            )
            summary_num_tokens = [
                encoded_summary.ne(self.tokenizer.pad_token_id).sum()
                for encoded_summary in encoded_summaries
            ]

            Batch = namedtuple(
                "Batch",
                [
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
                src=encoded_stories.to(device),
                segs=encoder_token_type_ids.to(device),
                mask_src=encoder_mask.to(device),
                tgt_num_tokens=torch.stack(summary_num_tokens).to(device),
                tgt=encoded_summaries.to(device),
                src_str=stories,
                tgt_str=summaries,
            )
        else:
            Batch = namedtuple("Batch", ["src", "segs", "mask_src"])
            batch = Batch(
                src=encoded_stories.to(device),
                segs=encoder_token_type_ids.to(device),
                mask_src=encoder_mask.to(device),
            )

        return batch

    def preprocess(self, story_lines, summary_lines=None):
        """preprocess multiple data points

           Args:
              story_lines (list of strings): List of sentences.
              targets (list of strings, optional): List of sentences.
                  Defaults to None, which means it doesn't include summary and is
                  not training data.

            Returns:
                If summary_lines is None, return list of list of token ids. Otherwise,
                return a tuple of (list of list of token ids, list of list of token ids).

        """
        story_lines_token_ids = []
        for line in story_lines:
            try:
                if len(line) <= 0:
                    continue
                story_lines_token_ids.append(
                    self.tokenizer.encode(line, max_length=self.max_src_len)
                )
            except:
                print(line)
                raise
        story_token_ids = [
            token for sentence in story_lines_token_ids for token in sentence
        ]
        if summary_lines:
            summary_lines_token_ids = []
            for line in summary_lines:
                try:
                    if len(line) <= 0:
                        continue
                    summary_lines_token_ids.append(
                        self.tokenizer.encode(line, max_length=self.max_tgt_len)
                    )
                except:
                    print(line)
                    raise
            summary_token_ids = [
                token for sentence in summary_lines_token_ids for token in sentence
            ]
            return story_token_ids, summary_token_ids
        else:
            return story_token_ids, None


def validate(summarizer, validate_dataset):
    """ validation function to be used optionally in fine tuning.

    Args:
        summarizer(BertSumAbs): The summarizer under fine tuning.
        validate_dataset (SummarizationDataset): dataset for validation.

    Returns:
        string: A string which contains the rouge score on a subset of
            the validation dataset.

    """
    TOP_N = 8
    shortened_dataset = validate_dataset.shorten(TOP_N)
    reference_summaries = [
        " ".join(t).rstrip("\n") for t in shortened_dataset.get_target()
    ]
    generated_summaries = summarizer.predict(
        shortened_dataset, num_gpus=1, batch_size=4
    )
    assert len(generated_summaries) == len(reference_summaries)
    print("###################")
    print("prediction is {}".format(generated_summaries[0]))
    print("reference is {}".format(reference_summaries[0]))

    rouge_score = compute_rouge_python(
        cand=generated_summaries, ref=reference_summaries
    )
    return "rouge score: {}".format(rouge_score)


class BertSumAbs(Transformer):
    """class which performs abstractive summarization fine tuning and
        prediction based on BertSumAbs model  """

    def __init__(
        self,
        processor,
        model_name="bert-base-uncased",
        finetune_bert=True,
        cache_dir=".",
        label_smoothing=0.1,
        test=False,
        max_pos_length=768,
    ):
        """Initialize an object of BertSumAbs.

        Args:
            processor (BertSumAbsProcessor): A processor with symbols, tokenizers
                and collate functions that are used in finetuning and prediction.
            model_name (str, optional:) Name of the pretrained model which is used
                to initialize the encoder of the BertSumAbs model.
                check MODEL_CLASS for supported models. Defaults to "bert-base-uncased".
            finetune_bert (bool, option): Whether the bert model in the encoder is
                finetune or not. Defaults to True.
            cache_dir (str, optional): Directory to cache the tokenizer.
                Defaults to ".".
            label_smoothing (float, optional): The amount of label smoothing.
                Value range is [0, 1]. Defaults to 0.1.
            test (bool, optional): Whether the class is initiated for test or not.
                It must be True if the class obj is only initialized to load a
                 checkpoint for test/inferencing.  Defaults to False.
            max_pos_length (int, optional): maximum postional embedding length for the
                input. Defaults to 768.
        """
        model = MODEL_CLASS[model_name].from_pretrained(
            model_name, cache_dir=cache_dir, num_labels=0, output_loading_info=False
        )
        super().__init__(model_name=model_name, model=model, cache_dir=cache_dir)

        if model_name not in self.list_supported_models():
            raise ValueError(
                "Model name {} is not supported by BertSumAbs. "
                "Call 'BertSumAbs.list_supported_models()' to get all supported model "
                "names.".format(value)
            )

        self.model_class = MODEL_CLASS[model_name]
        self.cache_dir = cache_dir
        self.max_pos_length = max_pos_length

        self.model = AbsSummarizer(
            temp_dir=cache_dir,
            finetune_bert=finetune_bert,
            checkpoint=None,
            label_smoothing=label_smoothing,
            symbols=processor.symbols,
            test=test,
            max_pos=self.max_pos_length,
        )
        self.processor = processor
        self.optim_bert = None
        self.optim_dec = None

    @staticmethod
    def list_supported_models():
        return list(MODEL_CLASS.keys())

    def fit(
        self,
        train_dataset,
        num_gpus=None,
        gpu_ids=None,
        batch_size=4,
        local_rank=-1,
        max_steps=5e4,
        warmup_steps_bert=20000,
        warmup_steps_dec=10000,
        learning_rate_bert=0.002,
        learning_rate_dec=0.2,
        optimization_method="adam",
        max_grad_norm=0,
        beta1=0.9,
        beta2=0.999,
        decay_method="noam",
        gradient_accumulation_steps=1,
        report_every=10,
        save_every=1000,
        verbose=True,
        seed=None,
        fp16=False,
        fp16_opt_level="O2",
        world_size=1,
        rank=0,
        validation_function=None,
        checkpoint=None,
        **kwargs,
    ):
        """
        Fine-tune pre-trained transofmer models for extractive summarization.

        Args:
            train_dataset (SummarizationDataset): Training dataset.
            num_gpus (int, optional): The number of GPUs to use. If None, all
                available GPUs will be used. If set to 0 or GPUs are not available,
                CPU device will be used. Defaults to None.
            gpu_ids (list): List of GPU IDs to be used.
                If set to None, the first num_gpus GPUs will be used.
                Defaults to None.
            batch_size (int, optional): Maximum number of tokens in each batch.
            local_rank (int, optional): Local_rank for distributed training on GPUs.
                Local rank means the ranking of the current GPU device on the current
                node. Defaults to -1, which means non-distributed training.
            max_steps (int, optional): Maximum number of training steps. Defaults to 5e5.
            warmup_steps_bert (int, optional): Number of steps taken to increase
                learning rate from 0 to `learning_rate` for tuning the BERT encoder.
                Defaults to 2e4.
            warmup_steps_dec (int, optional): Number of steps taken to increase
                learning rate from 0 to `learning_rate` for tuning the decoder.
                Defaults to 1e4.
            learning_rate_bert (float, optional):  Learning rate of the optimizer
                for the encoder. Defaults to 0.002.
            learning_rate_dec (float, optional):  Learning rate of the optimizer
                for the decoder. Defaults to 0.2.
            optimization_method (string, optional): Optimization method used in fine
                tuning. Defaults to "adam".
            max_grad_norm (float, optional): Maximum gradient norm for gradient clipping.
                Defaults to 0.
            beta1 (float, optional): The exponential decay rate for the first moment
                estimates. Defaults to 0.9.
            beta2 (float, optional): The exponential decay rate for the second-moment
                estimates. This value should be set close to 1.0 on problems with
                a sparse gradient. Defaults to 0.99.
            decay_method (string, optional): learning rate decrease method.
                Default to 'noam'.
            gradient_accumulation_steps (int, optional): Number of batches to accumulate
                gradients on between each model parameter update. Defaults to 1.
            report_every (int, optional): The interval by steps to print out the
                training log. Defaults to 10.
            save_every (int, optional): The interval by steps to save the finetuned 
                model. Defaults to 100.
            verbose (bool, optional): Whether to print out the training log.
                Defaults to True.
            seed (int, optional): Random seed used to improve reproducibility.
                Defaults to None.
            fp16 (bool, optional): Whether to use mixed precision training.
                Defaults to False.
            fp16_opt_level (str, optional): optimization level, refer to
                 https://nvidia.github.io/apex/amp.html#opt-levels for details.
                 Value choices are: "O0", "O1", "O2", "O3". Defaults to "O2".
            world_size (int, optional): Total number of GPUs that will be used.
                Defaults to 1.
            rank (int, optional): Global rank of the current GPU in distributed
                training. It's calculated with the rank of the current node in the
                cluster/world and the `local_rank` of the device in the current node.
                See an example in :file: `examples/text_summarization/
                abstractive_summarization_bertsum_cnndm_distributed_train.py`.
                Defaults to 0.
            validation_function (function, optional): function used in fitting to
                validate the performance. Default to None.
            checkpoint (str, optional): file path for a checkpoint based on which the
                training continues. Default to None.
        """

        # get device
        device, num_gpus = get_device(
            num_gpus=num_gpus, gpu_ids=gpu_ids, local_rank=local_rank
        )
        # move model to devices
        print("device is {}".format(device))
        if checkpoint:
            checkpoint = torch.load(checkpoint, map_location="cpu")
            self.model.load_checkpoint(checkpoint["model"])
        self.model = move_model_to_device(model=self.model, device=device)

        # init optimizer
        self.optim_bert = model_builder.build_optim_bert(
            self.model,
            optim=optimization_method,
            lr_bert=learning_rate_bert,
            warmup_steps_bert=warmup_steps_bert,
            max_grad_norm=max_grad_norm,
            beta1=beta1,
            beta2=beta2,
        )
        self.optim_dec = model_builder.build_optim_dec(
            self.model,
            optim=optimization_method,
            lr_dec=learning_rate_dec,
            warmup_steps_dec=warmup_steps_dec,
            max_grad_norm=max_grad_norm,
            beta1=beta1,
            beta2=beta2,
        )

        optimizers = [self.optim_bert, self.optim_dec]

        self.amp = get_amp(fp16)
        if self.amp:
            self.model, optim = self.amp.initialize(
                self.model, optimizers, opt_level=fp16_opt_level
            )

        global_step = 0
        if checkpoint:
            if checkpoint["optimizers"]:
                for i in range(len(optimizers)):
                    model_builder.load_optimizer_checkpoint(
                        optimizers[i], checkpoint["optimizers"][i]
                    )
            if self.amp and "amp" in checkpoint and checkpoint["amp"]:
                self.amp.load_state_dict(checkpoint["amp"])
            if "global_step" in checkpoint and checkpoint["global_step"]:
                global_step = checkpoint["global_step"] / world_size
                print("global_step is {}".format(global_step))

        self.model = parallelize_model(
            model=self.model,
            device=device,
            num_gpus=num_gpus,
            gpu_ids=gpu_ids,
            local_rank=local_rank,
        )

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
            train_dataset, sampler=sampler, batch_size=batch_size, collate_fn=collate_fn
        )

        # compute the max number of training steps
        max_steps = compute_training_steps(
            train_dataloader,
            max_steps=max_steps,
            gradient_accumulation_steps=gradient_accumulation_steps,
        )

        super().fine_tune(
            train_dataloader=train_dataloader,
            get_inputs=BertSumAbsProcessor.get_inputs,
            device=device,
            num_gpus=num_gpus,
            max_steps=max_steps,
            global_step=global_step,
            max_grad_norm=max_grad_norm,
            gradient_accumulation_steps=gradient_accumulation_steps,
            verbose=verbose,
            seed=seed,
            report_every=report_every,
            save_every=save_every,
            clip_grad_norm=False,
            optimizer=optimizers,
            scheduler=None,
            fp16=fp16,
            amp=self.amp,
            validation_function=validation_function,
        )

        # release GPU memories
        self.model.cpu()
        torch.cuda.empty_cache()

        self.save_model(max_steps)

    def predict(
        self,
        test_dataset,
        num_gpus=None,
        gpu_ids=None,
        local_rank=-1,
        batch_size=16,
        alpha=0.6,
        beam_size=5,
        min_length=15,
        max_length=150,
        fp16=False,
        verbose=True,
    ):
        """
        Predict the summarization for the input data iterator.

        Args:
            test_dataset (SummarizationDataset): Dataset for which the summary
                to be predicted.
            num_gpus (int, optional): The number of GPUs used in prediction.
                Defaults to 1.
            gpu_ids (list): List of GPU IDs to be used.
                If set to None, the first num_gpus GPUs will be used.
                Defaults to None.
            local_rank (int, optional): Local rank of the device in distributed
                inferencing. Defaults to -1, which means non-distributed inferencing.
            batch_size (int, optional): The number of test examples in each batch.
                Defaults to 16.
            alpha (float, optional): Length penalty. Defaults to 0.6.
            beam_size (int, optional): Beam size of beam search. Defaults to 5.
            min_length (int, optional): Minimum number of tokens in the output sequence.
                Defaults to 15.
            max_length (int, optional):  Maximum number of tokens in output
                sequence. Defaults to 150.
            fp16 (bool, optional): Whether to use half-precision model for prediction.
                Defaults to False.
            verbose (bool, optional): Whether to print out the training log.
                Defaults to True.

        Returns:
            List of strings which are the summaries

        """
        device, num_gpus = get_device(
            num_gpus=num_gpus, gpu_ids=gpu_ids, local_rank=local_rank
        )

        # move model to devices
        def this_model_move_callback(model, device):
            model = move_model_to_device(model, device)
            return parallelize_model(
                model, device, num_gpus=num_gpus, gpu_ids=gpu_ids, local_rank=local_rank
            )

        if fp16:
            self.model = self.model.half()

        self.model = move_model_to_device(self.model, device)
        self.model.eval()

        predictor = build_predictor(
            self.processor.tokenizer,
            self.processor.symbols,
            self.model,
            alpha=alpha,
            beam_size=beam_size,
            min_length=min_length,
            max_length=max_length,
        )
        predictor = this_model_move_callback(predictor, device)
        self.model = parallelize_model(
            self.model,
            device,
            num_gpus=num_gpus,
            gpu_ids=gpu_ids,
            local_rank=local_rank,
        )

        test_sampler = SequentialSampler(test_dataset)

        def collate_fn(data):
            return self.processor.collate(
                data, self.max_pos_length, device, train_mode=False
            )

        test_dataloader = DataLoader(
            test_dataset,
            sampler=test_sampler,
            batch_size=batch_size,
            collate_fn=collate_fn,
        )
        print("dataset length is {}".format(len(test_dataset)))

        def format_summary(translation):
            """ Transforms the output of the `from_batch` function
            into nicely formatted summaries.
            """
            raw_summary = translation
            summary = (
                raw_summary.replace("[unused0]", "")
                .replace("[unused3]", "")
                .replace("[CLS]", "")
                .replace("[SEP]", "")
                .replace("[PAD]", "")
                .replace("[unused1]", "")
                .replace(r" +", " ")
                .replace(" [unused2] ", ".")
                .replace("[unused2]", "")
                .strip()
            )

            return summary

        def generate_summary_from_tokenid(preds, pred_score):
            batch_size = preds.size()[0]  # batch.batch_size
            translations = []
            for b in range(batch_size):
                if len(preds[b]) < 1:
                    pred_sents = ""
                else:
                    pred_sents = self.processor.tokenizer.convert_ids_to_tokens(
                        [int(n) for n in preds[b] if int(n) != 0]
                    )
                    pred_sents = " ".join(pred_sents).replace(" ##", "")
                translations.append(pred_sents)
            return translations

        generated_summaries = []

        for batch in tqdm(
            test_dataloader, desc="Generating summary", disable=not verbose
        ):
            input = self.processor.get_inputs(batch, device, "bert", train_mode=False)
            translations, scores = predictor(**input)

            translations_text = generate_summary_from_tokenid(translations, scores)
            summaries = [format_summary(t) for t in translations_text]
            generated_summaries.extend(summaries)

        # release GPU memories
        self.model.cpu()
        torch.cuda.empty_cache()

        return generated_summaries

    def save_model(self, global_step=None, full_name=None):
        """
        save the trained model.

        Args:
            global_step (int, optional): The number of steps that the model has been
                finetuned for. Defaults to None.
            full_name (str, optional): File name to save the model's `state_dict()`.
                If it's None, the model is going to be saved under "fine_tuned" folder
                of the cached directory of the object. Defaults to None.
        """
        model_to_save = (
            self.model.module if hasattr(self.model, "module") else self.model
        )  # Take care of distributed/parallel training

        if full_name is None:
            output_model_dir = os.path.join(self.cache_dir, "fine_tuned")
            os.makedirs(self.cache_dir, exist_ok=True)
            os.makedirs(output_model_dir, exist_ok=True)
            full_name = os.path.join(output_model_dir, "bertsumabs.pt")
        else:
            path, filename = os.path.split(full_name)
            print(path)
            os.makedirs(path, exist_ok=True)

        checkpoint = {
            "optimizers": [self.optim_bert.state_dict(), self.optim_dec.state_dict()],
            "model": model_to_save.state_dict(),
            "amp": self.amp.state_dict() if self.amp else None,
            "global_step": global_step,
            "max_pos_length": self.max_pos_length,
        }

        logger.info("Saving model checkpoint to %s", full_name)
        try:
            print("saving through pytorch to {}".format(full_name))
            torch.save(checkpoint, full_name)
        except OSError:
            try:
                print("saving as pickle")
                pickle.dump(checkpoint, open(full_name, "wb"))
            except Exception:
                raise
        except Exception:
            raise
