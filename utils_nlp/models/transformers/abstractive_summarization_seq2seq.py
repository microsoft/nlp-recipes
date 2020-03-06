# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import json
import logging
from tqdm import tqdm
import random

import torch
from torch.utils.data import DataLoader, SequentialSampler, Dataset
from torch.utils.data.distributed import DistributedSampler

from transformers import RobertaConfig, BertConfig

from utils_nlp.models.transformers.common import TOKENIZER_CLASS, Transformer
from utils_nlp.common.pytorch_utils import (
    get_device,
    move_model_to_device,
    parallelize_model,
)
from s2s_ft.utils import (
    Seq2seqDatasetForBert,
    batch_list_to_batch_tensors,
)
from s2s_ft.modeling import BertForSequenceToSequence
from s2s_ft.modeling import UNILM_PRETRAINED_MODEL_ARCHIVE_MAP
from s2s_ft.tokenization_unilm import UnilmTokenizer
from s2s_ft.configuration_unilm import UnilmConfig, UNILM_PRETRAINED_CONFIG_ARCHIVE_MAP
from s2s_ft.config import BertForSeq2SeqConfig
import s2s_ft.s2s_loader as seq2seq_loader
from s2s_ft.modeling_decoding import BertForSeq2SeqDecoder

SUPPORTED_BERT_MODELS = ["bert-large-uncased", "bert-base-cased", "bert-large-cased"]
SUPPORTED_ROBERTA_MODELS = ["roberta-base", "roberta-large"]

# ROBERTA and XLM_ROBERTA are converted to BERT format by
# BertForSequenceToSequence.from_pretrained
MODEL_CLASS = {}
MODEL_CLASS.update({k: BertForSequenceToSequence for k in SUPPORTED_BERT_MODELS})
MODEL_CLASS.update({k: BertForSequenceToSequence for k in SUPPORTED_ROBERTA_MODELS})
MODEL_CLASS.update(
    {k: BertForSequenceToSequence for k in UNILM_PRETRAINED_MODEL_ARCHIVE_MAP}
)


TOKENIZER_CLASS.update({k: UnilmTokenizer for k in UNILM_PRETRAINED_CONFIG_ARCHIVE_MAP})

CONFIG_CLASS = {}
CONFIG_CLASS.update({k: BertConfig for k in SUPPORTED_BERT_MODELS})
CONFIG_CLASS.update({k: RobertaConfig for k in SUPPORTED_ROBERTA_MODELS})
CONFIG_CLASS.update({k: UnilmConfig for k in UNILM_PRETRAINED_CONFIG_ARCHIVE_MAP})

# XLM_ROBERTA is for multilingual and is WIP in s2s-ft.
# We can add it when it's finished and validated
# from transformers.modeling_xlm_roberta import XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
# MODEL_CLASS.update({k: BertForSequenceToSequence for k
# in XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP})
# CONFIG_CLASS.update({k: XLMRobertaConfig for k in
# XLM_ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP})


logger = logging.getLogger(__name__)


def _get_model_type(model_name):
    if "-".join(model_name.split("-")[:2]) == "xlm-roberta":
        return "xlm-roberta"
    elif model_name.startswith("unilm"):
        return "unilm"
    else:
        return model_name.split("-")[0]


def detokenize(tk_list):
    r_list = []
    for tk in tk_list:
        if tk.startswith("##") and len(r_list) > 0:
            r_list[-1] = r_list[-1] + tk[2:]
        else:
            r_list.append(tk)
    return r_list


class S2SAbsSumDataset(Dataset):
    """
    Dataset containing data processed and ready to be passed to
    S2SAbstractiveSummarizer.fit and S2SAbstractiveSummarizer.predict.
    """

    def __init__(self, features):
        self.features = features

    def __getitem__(self, idx):
        return self.features[idx]

    def __len__(self):
        return len(self.features)


class S2SAbsSumProcessor:
    """
    Processor with methods for converting input data in different formats
    to S2SAbsSumDataset for training and testing.

    Args:
        model_name (str, optional): Name of the model which determines the
            tokenizer to use. Call `S2SAbsSumProcessor.list_supported_models()`
            to see all supported model names. Defaults to "unilm-base-cased".
        to_lower (bool, optional): Whether to convert all letters to lower case
            during tokenization. This is determined by if a cased model is used.
            Defaults to False, which corresponds to a cased model.
        cache_dir (str, optional): Directory to cache the tokenizer.
            Defaults to ".".
    """

    def __init__(
        self, model_name="unilm-base-cased", to_lower=False, cache_dir=".",
    ):

        self.tokenizer = TOKENIZER_CLASS[model_name].from_pretrained(
            model_name, do_lower_case=to_lower, cache_dir=cache_dir
        )
        self.cache_dir = cache_dir
        self._model_name = model_name

    @staticmethod
    def list_supported_models():
        return list(MODEL_CLASS)

    @classmethod
    def get_inputs(cls, batch, device, model_name):
        """
        Converts a batch of features to model input format,
        used by Transformer.fine_tune.
        """
        batch = tuple(t.to(device) for t in batch)
        inputs = {
            "source_ids": batch[0],
            "target_ids": batch[1],
            "pseudo_ids": batch[2],
            "num_source_tokens": batch[3],
            "num_target_tokens": batch[4],
        }
        return inputs

    @staticmethod
    def create_s2s_dataset(
        examples,
        train_mode,
        tokenizer,
        output_dir,
        local_rank=-1,
        cached_features_file=None,
    ):
        """
        Creates S2SAbsSumDataset from input file or list of dictionaries.

        Args:
            examples (str or list): Input file path or list of dictionaries.
                The input file should be in the following format:
                {"src": "abcdefg", "tgt": "ag"}
                {"src": "hijklmn", "tgt": "hn"}
                where the "src" field is the input text to summarize and the "tgt"
                field is the summary.
                The list of dictionaries should be in similar format:
                [{"src": "abcdefg", "tgt": "ag"},
                {"src": "hijklmn", "tgt": "hn"}]
                The "tgt" field is optional if `train_mode` is False.
            train_mode (bool): Whether the input data is for training or testing.
                If True, both "src" and "tgt" fields need to be provided in
                `examples`.
                If False, only the "src" field is required.
            tokenizer (tokenizer): Tokenizer used to convert tokens to token ids. The
                type of the tokenizer depends on the model that will be used.
            output_dir (str): Directory to save the cached features files.
            local_rank (int, optional): Local rank of the device in distributed
                training. Defaults to -1, which means non-distributed training.
            cached_features_file (str, optional): Path of the cached features file.
                If provided and the file already exists, it is loaded and used.
                If provided and the file doesn't exist, processed features are
                saved to this file.
                If not provided, processed features are saved to `output_dir`.
                Defaults to None.

        Returns:
            S2SAbsSumDataset

        """
        if train_mode:
            cached_features_file_name = "cached_features_for_training.pt"
            shuffle_flag = True
        else:
            cached_features_file_name = "cached_features_for_testing.pt"
            shuffle_flag = False

        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        if cached_features_file is None:
            cached_features_file = os.path.join(output_dir, cached_features_file_name)
            if os.path.exists(cached_features_file):
                os.remove(cached_features_file)

        features = load_and_cache_examples(
            input_examples=examples,
            tokenizer=tokenizer,
            cached_features_file=cached_features_file,
            shuffle=shuffle_flag,
            local_rank=local_rank,
            train_mode=train_mode,
        )

        if not train_mode:
            features = [
                tokenizer.convert_ids_to_tokens(line["source_ids"]) for line in features
            ]

            features = sorted(list(enumerate(features)), key=lambda x: -len(x[1]))

        return S2SAbsSumDataset(features)

    def s2s_dataset_from_iterable_sum_ds(
        self, sum_ds, train_mode, cached_features_file=None, local_rank=-1
    ):
        """
        Converts IterableSummarizationDataset to S2SAbsSumDataset.

        Args:
            sum_ds (IterableSummarizationDataset): Input dataset.
            train_mode (bool): Whether the input data is for training or testing.
            cached_features_file (str, optional): Path of the cached features file.
                If provided and the file already exists, it is loaded and used.
                If provided and the file doesn't exist, processed features are
                saved to this file.
                If not provided, processed features are saved to cache_dir.
                Defaults to None.
            local_rank (int, optional): Local rank of the device in distributed
                training. Defaults to -1, which means non-distributed training.

        Returns:
            S2SAbsSumDataset
        """

        examples = []
        if train_mode:
            for source, target in zip(sum_ds, sum_ds.get_target()):
                examples.append({"src": source, "tgt": target})
        else:
            for source in sum_ds:
                examples.append({"src": source})

        s2s_dataset = S2SAbsSumProcessor.create_s2s_dataset(
            examples=examples,
            train_mode=train_mode,
            tokenizer=self.tokenizer,
            output_dir=self.cache_dir,
            local_rank=local_rank,
            cached_features_file=cached_features_file,
        )

        return s2s_dataset

    def s2s_dataset_from_sum_ds(
        self, sum_ds, train_mode, cached_features_file=None, local_rank=-1
    ):

        """
        Converts SummarizationDataset to S2SAbsSumDataset.

        Args:
            sum_ds (SummarizationDataset): Input dataset.
            train_mode (bool): Whether the input data is for training or testing.
            cached_features_file (str, optional): Path of the cached features file.
                If provided and the file already exists, it is loaded and used.
                If provided and the file doesn't exist, processed features are
                saved to this file.
                If not provided, processed features are saved to cache_dir.
                Defaults to None.
            local_rank (int, optional): Local rank of the device in distributed
                training. Defaults to -1, which means non-distributed training.

        Returns:
            S2SAbsSumDataset
        """
        examples = []
        for item in sum_ds:
            examples.append(item)

        s2s_dataset = S2SAbsSumProcessor.create_s2s_dataset(
            examples=examples,
            train_mode=train_mode,
            tokenizer=self.tokenizer,
            output_dir=self.cache_dir,
            local_rank=local_rank,
            cached_features_file=cached_features_file,
        )

        return s2s_dataset

    def s2s_dataset_from_json_or_file(
        self, input_data, train_mode, cached_features_file=None, local_rank=-1
    ):
        """
        Converts input file or list of dictionaries to S2SAbsSumDataset.

        Args:
            input_data (str or list): Input file path or list of dictionaries.
                The input file should be in the following format:
                {"src": "abcdefg", "tgt": "ag"}
                {"src": "hijklmn", "tgt": "hn"}
                where the "src" field is the input text to summarize and the "tgt"
                field is the summary.
                The list of dictionaries should be in similar format:
                [{"src": "abcdefg", "tgt": "ag"},
                {"src": "hijklmn", "tgt": "hn"}]
                The "tgt" field is optional if `train_mode` is False.
            train_mode (bool): Whether the input data is for training or testing.
            cached_features_file (str, optional): Path of the cached features file.
                If provided and the file already exists, it is loaded and used.
                If provided and the file doesn't exist, processed features are
                saved to this file.
                If not provided, processed features are saved to cache_dir.
                Defaults to None.
            local_rank (int, optional): Local rank of the device in distributed
                training. Defaults to -1, which means non-distributed training.

        Returns:
            S2SAbsSumDataset
        """

        s2s_dataset = S2SAbsSumProcessor.create_s2s_dataset(
            examples=input_data,
            train_mode=train_mode,
            tokenizer=self.tokenizer,
            output_dir=self.cache_dir,
            local_rank=local_rank,
            cached_features_file=cached_features_file,
        )

        return s2s_dataset


class S2SConfig:
    """
    This class contains some default decoding settings that the users usually
    don't need to change.

    Args:
        new_pos_ids (bool, optional): Whether to use new_pos_ids for LMs.
            Defaults to False.
        min_len (int, optional): Minimal length of the output.
            Defaults to 1.
        ngram_size (int, optional): Size of forbidden duplicate ngrams.
            Defaults to 3.
        mode (str, optional): Choose in "s2s" (sequence to sequence),
            "l2r" (left to right), and "both". Defaults to "s2s".
        s2s_special_token (bool, optional): If True, use a special cls token
            at the beginning of the sequence. Otherwise, use sep token at
            at the beginning of the sequence. Defaults to False.
        s2s_add_segment (bool, optional): If True, use special segment id for
            the first token. Otherwise, use the same segment id for the first
            token and the first sequence. Defaults to False.
        s2s_share_segment (bool, optional): If `s2s_add_segment=True` and
            `s2s_share_segement=True`, sharing segment embeddings for the
            encoder of S2S. Defaults to False.
        pos_shift (bool, optional): Whether to use position shift for
            fine-tuning. Defaults to False.


    """

    def __init__(
        self,
        new_pos_ids=False,
        min_len=1,
        ngram_size=3,
        mode="s2s",
        s2s_special_token=False,
        s2s_add_segment=False,
        s2s_share_segment=False,
        pos_shift=False,
        ffn_type=0,
        num_qkv=0,
        seg_emb=False,
    ):

        self.new_pos_ids = new_pos_ids
        self.min_len = min_len
        self.forbid_ngram_size = ngram_size
        self.mode = mode
        self.s2s_special_token = s2s_special_token
        self.s2s_add_segment = s2s_add_segment
        self.s2s_share_segment = s2s_share_segment
        self.pos_shift = pos_shift
        self.ffn_type = ffn_type
        self.num_qkv = num_qkv
        self.seg_emb = seg_emb

    def save_to_json(self, json_file):
        with open(json_file, "w") as f:
            json.dump(self.__dict__, f)

    @classmethod
    def load_from_json(cls, json_file):
        config = cls()
        with open(json_file, "r") as f:
            config.__dict__ = json.load(f)
        return config


class S2SAbstractiveSummarizer(Transformer):
    def __init__(
        self,
        model_name="unilm-base-cased",
        to_lower=False,
        cache_dir=".",
        load_model_from_dir=None,
        model_file_name=None,
        label_smoothing=0.1,
        max_seq_length=512,
        max_source_seq_length=464,
        max_target_seq_length=48,
    ):

        if model_name not in self.list_supported_models():
            raise ValueError(
                "Model name {0} is not supported by {1}. "
                "Call '{1}.list_supported_models()' to get all supported model "
                "names.".format(model_name, self.__class__.__name__)
            )
        model_class = MODEL_CLASS[model_name]
        config_class = CONFIG_CLASS[model_name]

        self._model_name = model_name
        self._model_type = _get_model_type(self._model_name)

        # self._bert_model_name is needed for BertForSeq2SeqDecoder
        if self._model_type != "bert":
            if self._model_type == "roberta":
                self._bert_model_name = (
                    self._model_name.replace("roberta", "bert") + "-cased"
                )
            else:
                self._bert_model_name = "bert-" + self._model_name.split("-", 1)[-1]
        else:
            self._bert_model_name = self._model_name

        self.cache_dir = cache_dir
        self.load_model_from_dir = load_model_from_dir
        self.do_lower_case = to_lower
        self.max_seq_length = max_seq_length
        self.max_source_seq_length = max_source_seq_length
        self.max_target_seq_length = max_target_seq_length

        if load_model_from_dir is None:
            model_to_load = self._model_name
        elif model_file_name is None:
            # Assume model was saved by
            # `:func:`~transformers.PreTrainedModel.save_pretrained``,
            # The load_model_from_dir should contain pytorch_model.bin and config.json
            # and can be loaded by
            # `:func:`~transformers.PreTrainedModel.from_pretrained``.
            logger.info("Loading cached model from {}".format(load_model_from_dir))
            model_to_load = load_model_from_dir
        else:
            # Assume model was saved by S2SAbstractiveSummarizer.save_model
            model_to_load = os.path.join(load_model_from_dir, model_file_name)
            logger.info("Loading cached model from {}".format(model_to_load))

        if load_model_from_dir is not None and model_file_name is None:
            # Assume config.json is in load_model_from_dir
            model_config = config_class.from_pretrained(
                load_model_from_dir, cache_dir=cache_dir
            )
        else:
            model_config = config_class.from_pretrained(
                self._model_name, cache_dir=cache_dir
            )

        # Convert regular model config to sequence to sequence config
        config = BertForSeq2SeqConfig.from_exist_config(
            config=model_config,
            label_smoothing=label_smoothing,
            max_position_embeddings=self.max_source_seq_length
            + self.max_target_seq_length,
        )
        logger.info("Model config for seq2seq: %s", str(config))

        self.model = model_class.from_pretrained(
            model_to_load,
            config=config,
            model_type=self._model_type,
            cache_dir=cache_dir,
            reuse_position_embedding=True,
        )

        self.tokenizer = TOKENIZER_CLASS[model_name].from_pretrained(
            self._model_name,
            do_lower_case=to_lower,
            cache_dir=cache_dir,
            output_loading_info=False,
        )

    @staticmethod
    def list_supported_models():
        return list(MODEL_CLASS)

    def fit(
        self,
        train_dataset,
        learning_rate=5e-5,
        per_gpu_batch_size=8,
        num_epochs=1,
        recover_step=-1,
        recover_dir=None,
        max_steps=-1,
        local_rank=-1,
        num_gpus=None,
        gpu_ids=None,
        gradient_accumulation_steps=1,
        weight_decay=0.01,
        adam_epsilon=1e-8,
        warmup_steps=0,
        fp16=False,
        fp16_opt_level="O1",
        max_grad_norm=1.0,
        save_model_to_dir=None,
        verbose=True,
        seed=None,
        random_prob=0.1,
        keep_prob=0.1,
    ):
        global_step = 0
        if recover_step > 0:
            model_recover_checkpoint = os.path.join(
                recover_dir, "model.{}.bin".format(recover_step)
            )
            logger.info(
                " ** Recover model checkpoint in %s ** ", model_recover_checkpoint
            )
            model_state_dict = torch.load(model_recover_checkpoint, map_location="cpu")
            optimizer_recover_checkpoint = os.path.join(
                recover_dir, "optim.{}.bin".format(recover_step)
            )
            checkpoint_state_dict = torch.load(
                optimizer_recover_checkpoint, map_location="cpu"
            )

            checkpoint_state_dict["model"] = model_state_dict
            global_step = recover_step
        else:
            checkpoint_state_dict = None

        device, num_gpus, amp = self.prepare_model_and_optimizer(
            num_gpus=num_gpus,
            gpu_ids=gpu_ids,
            local_rank=local_rank,
            fp16=fp16,
            fp16_opt_level=fp16_opt_level,
            weight_decay=weight_decay,
            learning_rate=learning_rate,
            adam_epsilon=adam_epsilon,
            checkpoint_state_dict=checkpoint_state_dict,
        )

        per_node_train_batch_size = (
            per_gpu_batch_size * max(1, num_gpus) * gradient_accumulation_steps
        )

        # actual batch size, i.e. number of samples between each parameter update
        batch_size = per_node_train_batch_size * (
            torch.distributed.get_world_size() if local_rank != -1 else 1
        )

        # max_steps is mainly used by the scheduler to determine the learning rate,
        # together with global_step
        if max_steps == -1:
            max_steps = max(num_epochs * len(train_dataset) // batch_size, 1)

        if max_steps <= global_step:
            logger.info("Training is done. Please use a new dir or clean this dir!")

            return

        self.scheduler = Transformer.get_default_scheduler(
            optimizer=self.optimizer,
            warmup_steps=warmup_steps,
            num_training_steps=max_steps,
        )
        if recover_step > 0:
            self.scheduler.load_state_dict(checkpoint_state_dict["lr_scheduler"])

        train_dataset = Seq2seqDatasetForBert(
            features=train_dataset,
            max_source_len=self.max_source_seq_length,
            max_target_len=self.max_target_seq_length,
            vocab_size=self.tokenizer.vocab_size,
            cls_id=self.tokenizer.cls_token_id,
            sep_id=self.tokenizer.sep_token_id,
            pad_id=self.tokenizer.pad_token_id,
            mask_id=self.tokenizer.mask_token_id,
            random_prob=random_prob,
            keep_prob=keep_prob,
            num_training_instances=batch_size * max_steps,
            offset=batch_size * global_step,
        )

        # The training features are shuffled
        train_sampler = (
            SequentialSampler(train_dataset)
            if local_rank == -1
            else DistributedSampler(train_dataset, shuffle=False)
        )
        # batch_size of the dataloader is the number of samples to load each
        # iteration on each node
        train_dataloader = DataLoader(
            train_dataset,
            sampler=train_sampler,
            batch_size=per_node_train_batch_size // gradient_accumulation_steps,
            collate_fn=batch_list_to_batch_tensors,
        )

        global_step, _ = super().fine_tune(
            train_dataloader=train_dataloader,
            device=device,
            num_gpus=num_gpus,
            get_inputs=S2SAbsSumProcessor.get_inputs,
            max_steps=max_steps,
            global_step=global_step,
            gradient_accumulation_steps=gradient_accumulation_steps,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            local_rank=local_rank,
            fp16=fp16,
            amp=amp,
            max_grad_norm=max_grad_norm,
            verbose=verbose,
            seed=seed,
        )

        if save_model_to_dir is not None and local_rank in [-1, 0]:
            self.save_model(save_model_to_dir, global_step, fp16)

        # release GPU memories
        self.model.cpu()
        torch.cuda.empty_cache()

    def predict(
        self,
        test_dataset,
        per_gpu_batch_size=4,
        max_tgt_length=128,
        beam_size=1,
        need_score_traces=False,
        length_penalty=0,
        forbid_duplicate_ngrams=True,
        forbid_ignore_word=".",
        s2s_config=S2SConfig(),
        num_gpus=None,
        gpu_ids=None,
        local_rank=-1,
        fp16=False,
        verbose=True,
    ):
        if need_score_traces and beam_size <= 1:
            raise ValueError(
                "Score trace is only available for beam search with beam size > 1."
            )
        if max_tgt_length >= self.max_seq_length - 2:
            raise ValueError("Maximum tgt length exceeds max seq length - 2.")

        # preprocessing pipeline
        if self._model_type == "roberta":
            is_roberta = True
            no_segment_embedding = True
            vocab = self.tokenizer.encoder
        else:
            is_roberta = False
            no_segment_embedding = False
            vocab = self.tokenizer.vocab

        if not self._model_name.startswith("unilm1.2"):
            if self._model_name.startswith("unilm-") or self._model_name.startswith(
                "unilm1-"
            ):
                new_segment_ids = True
            else:
                new_segment_ids = False
        else:
            new_segment_ids = False

        cls_token = "<s>" if is_roberta else "[CLS]"
        sep_token = "</s>" if is_roberta else "[SEP]"
        pad_token = "<pad>" if is_roberta else "[PAD]"
        mask_token = "<mask>" if is_roberta else "[MASK]"

        max_src_length = self.max_seq_length - 2 - max_tgt_length
        bi_uni_pipeline = []
        bi_uni_pipeline.append(
            seq2seq_loader.Preprocess4Seq2seqDecoder(
                list(vocab.keys()),
                self.tokenizer.convert_tokens_to_ids,
                self.max_seq_length,
                max_tgt_length=max_tgt_length,
                new_segment_ids=new_segment_ids,
                mode=s2s_config.mode,
                num_qkv=s2s_config.num_qkv,
                s2s_special_token=s2s_config.s2s_special_token,
                s2s_add_segment=s2s_config.s2s_add_segment,
                s2s_share_segment=s2s_config.s2s_share_segment,
                pos_shift=s2s_config.pos_shift,
                cls_token=cls_token,
                sep_token=sep_token,
                pad_token=pad_token,
            )
        )

        def collate_fn(input_batch):
            buf_id = [x[0] for x in input_batch]
            buf = [x[1][:max_src_length] for x in input_batch]
            max_a_len = max([len(x) for x in buf])
            instances = []
            for instance in [(x, max_a_len) for x in buf]:
                for proc in bi_uni_pipeline:
                    instance = proc(instance)
                instances.append(instance)
            batch = seq2seq_loader.batch_list_to_batch_tensors(instances)

            return (batch, buf_id)

        # prepare decoder
        pair_num_relation = 0
        cls_num_labels = 2
        type_vocab_size = (
            6 + (1 if s2s_config.s2s_add_segment else 0) if new_segment_ids else 2
        )
        (
            mask_word_id,
            eos_word_ids,
            sos_word_id,
        ) = self.tokenizer.convert_tokens_to_ids([mask_token, sep_token, sep_token])
        forbid_ignore_set = None
        if forbid_ignore_word:
            w_list = []
            for w in forbid_ignore_word.split("|"):
                if w.startswith("[") and w.endswith("]"):
                    w_list.append(w.upper())
                else:
                    w_list.append(w)
            forbid_ignore_set = set(self.tokenizer.convert_tokens_to_ids(w_list))

        if hasattr(self.model, "module"):
            state_dict = self.model.module.state_dict()
        else:
            state_dict = self.model.state_dict()

        model = BertForSeq2SeqDecoder.from_pretrained(
            self._bert_model_name,
            state_dict=state_dict,
            num_labels=cls_num_labels,
            num_rel=pair_num_relation,
            type_vocab_size=type_vocab_size,
            task_idx=3,
            mask_word_id=mask_word_id,
            search_beam_size=beam_size,
            length_penalty=length_penalty,
            eos_id=eos_word_ids,
            sos_id=sos_word_id,
            forbid_duplicate_ngrams=forbid_duplicate_ngrams,
            forbid_ignore_set=forbid_ignore_set,
            ngram_size=s2s_config.forbid_ngram_size,
            min_len=s2s_config.min_len,
            mode=s2s_config.mode,
            max_position_embeddings=self.max_seq_length,
            ffn_type=s2s_config.ffn_type,
            num_qkv=s2s_config.num_qkv,
            seg_emb=s2s_config.seg_emb,
            pos_shift=s2s_config.pos_shift,
            is_roberta=is_roberta,
            no_segment_embedding=no_segment_embedding,
        )

        del state_dict

        if fp16:
            model.half()
        # get device
        device, num_gpus = get_device(
            num_gpus=num_gpus, gpu_ids=gpu_ids, local_rank=local_rank
        )

        # # move model
        model = move_model_to_device(model=model, device=device)

        batch_size = per_gpu_batch_size * max(1, num_gpus)

        model = parallelize_model(
            model=model,
            device=device,
            num_gpus=num_gpus,
            gpu_ids=gpu_ids,
            local_rank=local_rank,
        )

        # torch.cuda.empty_cache()
        model.eval()
        first_batch = True
        batch_count = 0

        output_lines = [""] * len(test_dataset)
        score_trace_list = [None] * len(test_dataset)

        test_sampler = SequentialSampler(test_dataset)
        test_dataloader = DataLoader(
            test_dataset,
            sampler=test_sampler,
            batch_size=batch_size,
            collate_fn=collate_fn,
        )
        for batch, buf_id in tqdm(
            test_dataloader, desc="Evaluating", disable=not verbose
        ):
            batch_count += 1
            with torch.no_grad():
                batch = [t.to(device) if t is not None else None for t in batch]
                (
                    input_ids,
                    token_type_ids,
                    position_ids,
                    input_mask,
                    mask_qkv,
                    task_idx,
                ) = batch
                traces = model(
                    input_ids,
                    token_type_ids,
                    position_ids,
                    input_mask,
                    task_idx=task_idx,
                    mask_qkv=mask_qkv,
                )
                if beam_size > 1:
                    traces = {k: v.tolist() for k, v in traces.items()}
                    output_ids = traces["pred_seq"]
                else:
                    output_ids = traces.tolist()

                for i in range(len(batch[0])):
                    w_ids = output_ids[i]
                    output_buf = self.tokenizer.convert_ids_to_tokens(w_ids)
                    output_tokens = []
                    for t in output_buf:
                        if t in (sep_token, pad_token):
                            break
                        output_tokens.append(t)
                    if is_roberta:
                        output_sequence = self.tokenizer.convert_tokens_to_string(
                            output_tokens
                        )
                    else:
                        output_sequence = " ".join(detokenize(output_tokens))
                    if "\n" in output_sequence:
                        output_sequence = " [X_SEP] ".join(output_sequence.split("\n"))
                    output_lines[buf_id[i]] = output_sequence
                    if first_batch or batch_count % 50 == 0:
                        logger.info("{} = {}".format(buf_id[i], output_sequence))
                    if need_score_traces:
                        score_trace_list[buf_id[i]] = {
                            "scores": traces["scores"][i],
                            "wids": traces["wids"][i],
                            "ptrs": traces["ptrs"][i],
                        }

            first_batch = False

        del model
        del batch
        torch.cuda.empty_cache()

        if need_score_traces:
            return output_lines, score_trace_list
        else:
            return output_lines

    def save_model(self, output_dir, global_step, fp16):
        model_to_save = (
            self.model.module if hasattr(self.model, "module") else self.model
        )
        torch.save(
            model_to_save.state_dict(),
            os.path.join(output_dir, "model.{}.bin".format(global_step)),
        )
        optim_to_save = {
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.scheduler.state_dict(),
        }
        if fp16:
            optim_to_save["amp"] = self.amp_state_dict
        torch.save(
            optim_to_save, os.path.join(output_dir, "optim.{}.bin".format(global_step)),
        )


def load_and_cache_examples(
    input_examples,
    tokenizer,
    local_rank,
    train_mode=True,
    cached_features_file=None,
    shuffle=True,
):

    # Make sure only the first process in distributed training process the dataset,
    # and the others will use the cache
    if local_rank not in [-1, 0]:
        torch.distributed.barrier()

    if cached_features_file is not None and os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        if isinstance(input_examples, str):
            logger.info("Creating features from dataset file at %s", input_examples)
            examples = []
            with open(input_examples, mode="r", encoding="utf-8") as reader:
                for line in reader:
                    examples.append(json.loads(line))
        else:
            examples = input_examples

        features = []
        if train_mode:
            for example in tqdm(examples):
                if isinstance(example["src"], list):
                    source_tokens = example["src"]
                    target_tokens = example["tgt"]
                else:
                    source_tokens = tokenizer.tokenize(example["src"])
                    target_tokens = tokenizer.tokenize(example["tgt"])
                features.append(
                    {
                        "source_ids": tokenizer.convert_tokens_to_ids(source_tokens),
                        "target_ids": tokenizer.convert_tokens_to_ids(target_tokens),
                    }
                )
        else:
            for example in tqdm(examples):
                if isinstance(example["src"], list):
                    source_tokens = example["src"]
                else:
                    source_tokens = tokenizer.tokenize(example["src"])
                features.append(
                    {"source_ids": tokenizer.convert_tokens_to_ids(source_tokens),}
                )

        if shuffle:
            random.shuffle(features)

        if cached_features_file is not None:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    # Make sure only the first process in distributed training process the dataset,
    # and the others will use the cache
    if local_rank == 0:
        torch.distributed.barrier()

    return features
