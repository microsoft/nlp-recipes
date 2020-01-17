import os
import tempfile
import shutil
import jsonlines
import logging

import torch
from torch.utils.data import DataLoader, SequentialSampler, Dataset
from torch.utils.data.distributed import DistributedSampler

from transformers.modeling_roberta import ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
from transformers.modeling_bert import BERT_PRETRAINED_MODEL_ARCHIVE_MAP
from transformers.modeling_xlm_roberta import XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
from transformers import RobertaConfig, BertConfig, DistilBertConfig, XLMRobertaConfig
from transformers import (
    BERT_PRETRAINED_CONFIG_ARCHIVE_MAP,
    ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP,
    DISTILBERT_PRETRAINED_CONFIG_ARCHIVE_MAP,
    XLM_ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP,
)


from utils_nlp.models.transformers.common import TOKENIZER_CLASS, Transformer
from utils_nlp.common.pytorch_utils import get_device, move_model_to_device
from s2s_ft.utils import load_and_cache_examples, Seq2seqDatasetForBert, batch_list_to_batch_tensors
from s2s_ft.modeling import BertForSequenceToSequence
from s2s_ft.modeling import UNILM_PRETRAINED_MODEL_ARCHIVE_MAP
from s2s_ft.tokenization_unilm import UnilmTokenizer
from s2s_ft.configuration_unilm import UnilmConfig, UNILM_PRETRAINED_CONFIG_ARCHIVE_MAP
from s2s_ft.config import BertForSeq2SeqConfig


MODEL_CLASS = {}
MODEL_CLASS.update({k: BertForSequenceToSequence for k in BERT_PRETRAINED_MODEL_ARCHIVE_MAP})
MODEL_CLASS.update({k: BertForSequenceToSequence for k in ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP})
MODEL_CLASS.update({k: BertForSequenceToSequence for k in XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP})
MODEL_CLASS.update({k: BertForSequenceToSequence for k in UNILM_PRETRAINED_MODEL_ARCHIVE_MAP})

TOKENIZER_CLASS.update({k: UnilmTokenizer for k in UNILM_PRETRAINED_CONFIG_ARCHIVE_MAP})

CONFIG_CLASS = {}
CONFIG_CLASS.update({k: BertConfig for k in BERT_PRETRAINED_CONFIG_ARCHIVE_MAP})
CONFIG_CLASS.update({k: RobertaConfig for k in ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP})
CONFIG_CLASS.update({k: DistilBertConfig for k in DISTILBERT_PRETRAINED_CONFIG_ARCHIVE_MAP})
CONFIG_CLASS.update({k: XLMRobertaConfig for k in XLM_ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP})
CONFIG_CLASS.update({k: UnilmConfig for k in UNILM_PRETRAINED_CONFIG_ARCHIVE_MAP})

logger = logging.getLogger(__name__)


class S2SAbsSumTrainDataset(Dataset):
    def __init__(self, train_features):
        self.train_features = train_features

    def __getitem__(self, idx):
        return self.train_features[idx]

    def __len__(self):
        return len(self.train_features)


class S2SAbsSumProcessor:
    def __init__(self, model_name="bert-base-cased", to_lower=False, cache_dir="."):
        self.tokenizer = TOKENIZER_CLASS[model_name].from_pretrained(
            model_name, do_lower_case=to_lower, cache_dir=cache_dir, output_loading_info=False
        )

        self.cached_features_file = os.path.join(cache_dir, "train_features")

    @classmethod
    def get_inputs(batch, device, model_name, train_model=True):
        batch = tuple(t.to(device) for t in batch)
        inputs = {
            "source_ids": batch[0],
            "target_ids": batch[1],
            "pseudo_ids": batch[2],
            "num_source_tokens": batch[3],
            "num_target_tokens": batch[4],
        }
        return inputs

    def train_dataset_from_iterable_sum_ds(self, sum_ds, local_rank=-1, keep_train_file=False):
        if local_rank not in [-1, 0]:
            torch.distributed.barrier()

        temp_dir = tempfile.mkdtemp()
        temp_train_file = os.path.join(temp_dir, "train_file.jsonl")
        try:
            with jsonlines.open(temp_train_file, mode="w") as writer:
                for source, target in zip(sum_ds, sum_ds.get_target()):
                    writer.write({"src": source, "tgt": target})

            train_dataset = self.train_dataset_from_file(train_file=temp_train_file, local_rank=-1)

        finally:
            if not keep_train_file:
                if os.path.isdir(temp_dir):
                    shutil.rmtree(temp_dir)
        if local_rank == 0:
            torch.distributed.barrier()
        return train_dataset

    def train_dataset_from_sum_ds(self, sum_ds, local_rank=-1, keep_train_file=False):
        if local_rank not in [-1, 0]:
            torch.distributed.barrier()

        temp_dir = tempfile.mkdtemp()
        temp_train_file = os.path.join(temp_dir, "train_file.jsonl")
        try:
            with jsonlines.open(temp_train_file, mode="w") as writer:
                for item in sum_ds:
                    writer.write(item)
            train_dataset = self.train_dataset_from_file(train_file=temp_train_file, local_rank=-1)

        finally:
            if not keep_train_file:
                if os.path.isdir(temp_dir):
                    shutil.rmtree(temp_dir)
        if local_rank == 0:
            torch.distributed.barrier()
        return train_dataset

    def train_dataset_from_file(self, train_file, local_rank=-1):

        train_features = load_and_cache_examples(
            example_file=train_file,
            tokenizer=self.tokenizer,
            local_rank=local_rank,
            cached_features_file=self.cached_features_file,
            shuffle=True,
        )

        return S2SAbsSumTrainDataset(train_features)


class S2SAbstractiveSummarizer(Transformer):
    def __init__(
        self,
        model_name="unilm-base-cased",
        to_lower=False,
        cache_dir=".",
        load_model_from_dir=None,
        label_smoothing=0.1,
        *model_args,
        **kwargs
    ):

        if model_name not in self.list_supported_models():
            raise ValueError(
                "Model name {0} is not supported by {1}. "
                "Call '{1}.list_supported_models()' to get all supported model "
                "names.".format(value, self.__class__.__name__)
            )
        model_class = MODEL_CLASS[model_name]
        config_class = CONFIG_CLASS[model_name]

        self._model_name = model_name
        self.cache_dir = cache_dir
        self.load_model_from_dir = load_model_from_dir

        if "-".join(model_name.split("-")[:2]) == "xlm-roberta":
            self._model_type = "xlm-roberta"
        else:
            self._model_type = model_name.split("-")[0]

        if load_model_from_dir is None:
            model_to_load = self._model_name
        else:
            logger.info("Loading cached model from {}".format(load_model_from_dir))
            model_to_load = load_model_from_dir

        model_config = config_class.from_pretrained(model_to_load, cache_dir=cache_dir)
        config = BertForSeq2SeqConfig.from_exist_config(
            config=model_config, label_smoothing=label_smoothing
        )
        logger.info("Model config for seq2seq: %s", str(config))

        self.model = model_class.from_pretrained(
            model_to_load, config=config, model_type=self._model_type, cache_dir=cache_dir
        )

        self.tokenizer = TOKENIZER_CLASS[model_name].from_pretrained(
            model_to_load, do_lower_case=to_lower, cache_dir=cache_dir, output_loading_info=False
        )

    @staticmethod
    def list_supported_models():
        return list(MODEL_CLASS)

    def fit(
        self,
        train_dataset,
        max_source_seq_length=464,
        max_target_seq_length=48,
        learning_rate=5e-5,
        batch_size=8,
        num_epochs=1,
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
        verbose=True,
        seed=None,
        random_prob=0.1,
        keep_prob=0.1,
    ):

        if gpu_ids is not None:
            per_node_train_batch_size = batch_size * len(gpu_ids)
        elif num_gpus is not None:
            per_node_train_batch_size = batch_size * num_gpus
        else:
            per_node_train_batch_size = batch_size * max(1, torch.cuda.device_count())

        per_node_train_batch_size = per_node_train_batch_size * gradient_accumulation_steps

        batch_size = per_node_train_batch_size * (
            torch.distributed.get_world_size() if local_rank != -1 else 1
        )

        if max_steps == -1:
            max_steps = num_epochs * len(train_dataset) // batch_size

        # get device
        device, num_gpus = get_device(num_gpus=num_gpus, local_rank=local_rank)
        # move model
        self.model = move_model_to_device(self.model, device, num_gpus, gpu_ids, local_rank)

        train_dataset = Seq2seqDatasetForBert(
            features=train_dataset,
            max_source_len=max_source_seq_length,
            max_target_len=max_target_seq_length,
            vocab_size=self.tokenizer.vocab_size,
            cls_id=self.tokenizer.cls_token_id,
            sep_id=self.tokenizer.sep_token_id,
            pad_id=self.tokenizer.pad_token_id,
            mask_id=self.tokenizer.mask_token_id,
            random_prob=random_prob,
            keep_prob=keep_prob,
            num_training_instances=batch_size * max_steps,
            offset=0,
        )

        # init optimizer and scheduler
        self.optimizer = Transformer.get_default_optimizer(
            self.model, weight_decay, learning_rate, adam_epsilon
        )

        # TODO: Double check this part
        train_sampler = (
            SequentialSampler(train_dataset)
            if local_rank == -1
            else DistributedSampler(train_dataset, shuffle=False)
        )
        train_dataloader = DataLoader(
            train_dataset,
            sampler=train_sampler,
            batch_size=batch_size // gradient_accumulation_steps,
            collate_fn=batch_list_to_batch_tensors,
        )

        self.scheduler = Transformer.get_default_scheduler(
            optimizer=self.optimizer, warmup_steps=warmup_steps, num_training_steps=max_steps
        )

        super().fine_tune(
            train_dataloader=train_dataloader,
            device=device,
            num_gpus=num_gpus,
            get_inputs=S2SAbsSumProcessor.get_inputs,
            max_steps=max_steps,
            gradient_accumulation_steps=gradient_accumulation_steps,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            local_rank=local_rank,
            fp16=fp16,
            fp16_opt_level=fp16_opt_level,
            max_grad_norm=max_grad_norm,
            verbose=verbose,
            seed=seed,
        )

    def predict(self):
        pass

    def save_model(self, global_step, fp16):
        model_to_save = self.model.module if hasattr(self.model, "module") else self.model
        torch.save(
            model_to_save.state_dict(),
            os.path.join(self.cache_dir, "model.{}.bin".format(global_step)),
        )
        optim_to_save = {
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.scheduler.state_dict(),
        }
        if fp16:
            optim_to_save["amp"] = self.amp_state_dict
        torch.save(optim_to_save, os.path.join(self.cache_dir, "optim.{}.bin".format(global_step)))
