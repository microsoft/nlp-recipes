# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import torch
from torch.utils.data import TensorDataset

from pytorch_transformers.modeling_bert import (
    BERT_PRETRAINED_MODEL_ARCHIVE_MAP,
    BertForSequenceClassification,
)
from pytorch_transformers.modeling_distilbert import (
    DISTILBERT_PRETRAINED_MODEL_ARCHIVE_MAP,
    DistilBertForSequenceClassification,
)
from pytorch_transformers.modeling_roberta import (
    ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP,
    RobertaForSequenceClassification,
)
from pytorch_transformers.modeling_xlnet import (
    XLNET_PRETRAINED_MODEL_ARCHIVE_MAP,
    XLNetForSequenceClassification,
)

from utils_nlp.models.transformers.common import MAX_SEQ_LEN, TOKENIZER_CLASS, fine_tune

MODEL_CLASS = {}
MODEL_CLASS.update({k: BertForSequenceClassification for k in BERT_PRETRAINED_MODEL_ARCHIVE_MAP})
MODEL_CLASS.update(
    {k: RobertaForSequenceClassification for k in ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP}
)
MODEL_CLASS.update({k: XLNetForSequenceClassification for k in XLNET_PRETRAINED_MODEL_ARCHIVE_MAP})
MODEL_CLASS.update(
    {k: DistilBertForSequenceClassification for k in DISTILBERT_PRETRAINED_MODEL_ARCHIVE_MAP}
)


class SupportedModels:
    """Generic Class to be inherited by classes that want to support PyTorch Transformers
    
    Returns:
        list -- list of supported PyTorch Transformers
    """

    @staticmethod
    def list_supported_models():
        return list(MODEL_CLASS)


class Processor(SupportedModels):
    def __init__(self, model_name="bert-base-cased", tokenize=None, to_lower=False, cache_dir="."):
        self.tokenizer = TOKENIZER_CLASS[model_name].from_pretrained(
            model_name, do_lower_case=to_lower, cache_dir=cache_dir
        )
        self.custom_tokenize = tokenize

    @staticmethod
    def get_inputs(batch, model_name):
        if model_name.split("-")[0] == "bert":
            return {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[2]}
        else:
            raise ValueError("Model not supported.")

    def preprocess(self, text, labels, max_len):
        """preprocess data or batches"""
        if max_len > MAX_SEQ_LEN:
            print("setting max_len to max allowed sequence length: {}".format(MAX_SEQ_LEN))
            max_len = MAX_SEQ_LEN

        if self.custom_tokenize:
            tokens = [self.custom_tokenize(x) for x in text]
        else:
            tokens = [self.tokenizer.tokenize(x) for x in text]

        # truncate and add CLS & SEP markers
        tokens = [["[CLS]"] + x[0 : max_len - 2] + ["[SEP]"] for x in tokens]
        # get input ids
        input_ids = [self.tokenizer.convert_tokens_to_ids(x) for x in tokens]
        # pad sequence
        input_ids = [x + [0] * (max_len - len(x)) for x in input_ids]
        # create input mask
        input_mask = [[min(1, x) for x in y] for y in input_ids]
        # create segment ids
        # segment_ids = None
        td = TensorDataset(
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(input_mask, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
        )
        return td


class SequenceClassifier(SupportedModels):
    def __init__(self, model_name="bert-base-cased", num_labels=2, cache_dir=".", seed=0):
        self.model_name = model_name
        self.model = MODEL_CLASS[model_name].from_pretrained(
            model_name, cache_dir=cache_dir, num_labels=num_labels
        )
        self.seed = seed

    def fit(
        self,
        train_dataset,
        device="cuda",
        num_epochs=1,
        batch_size=32,
        num_gpus=None,
        local_rank=-1,
    ):
        # get device
        if local_rank == -1:
            device = torch.device(
                "cuda" if torch.cuda.is_available() and device == "cuda" else "cpu"
            )
            num_gpus = (
                min(num_gpus, torch.cuda.device_count()) if num_gpus else torch.cuda.device_count()
            )
        else:
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
            torch.distributed.init_process_group(backend="nccl")
            num_gpus = 1

        self.model.to(device)

        fine_tune(
            model=self.model,
            model_name=self.model_name,
            train_dataset=train_dataset,
            get_inputs=Processor.get_inputs,
            device=device,
            max_steps=-1,
            num_train_epochs=num_epochs,
            gradient_accumulation_steps=1,
            per_gpu_train_batch_size=batch_size,
            n_gpu=num_gpus,
            weight_decay=0.0,
            learning_rate=5e-5,
            adam_epsilon=1e-8,
            warmup_steps=0,
            fp16=False,
            fp16_opt_level="O1",
            local_rank=-1,
            seed=self.seed,
        )

    def predict():
        pass

