# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.


from enum import Enum, auto
import numpy as np
import torch.nn as nn
import torch
from pytorch_pretrained_bert.modeling import BertForSequenceClassification
from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.tokenization import BertTokenizer
from utils_nlp.pytorch.device import get_device
from utils_nlp.dataset.batch_utils import get_batch_rnd


class Language(Enum):
    ENGLISH = "bert-base-uncased"
    CHINESE = "bert-base-chinese"
    SPANISH = auto()
    HINDI = auto()
    FRENCH = auto()


class BERTSequenceClassifier:

    BERT_MAX_LEN = 512

    def __init__(
        self, pretrained_model=Language.ENGLISH, num_labels=2, cache_dir="."
    ):

        self.pretrained_model = pretrained_model
        self.num_labels = num_labels
        self._max_len = BERTSequenceClassifier.BERT_MAX_LEN
        self.cache_dir = cache_dir
        # create tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(
            pretrained_model.value, do_lower_case=False, cache_dir=cache_dir
        )
        # create classifier
        self.model = BertForSequenceClassification.from_pretrained(
            pretrained_model.value, cache_dir=cache_dir, num_labels=num_labels
        )
        self._is_trained = False

    def _get_tokens(self, text):
        tokens = [
            self.tokenizer.tokenize(x)[0 : self._max_len - 2] for x in text
        ]
        tokens = [["[CLS]"] + x + ["[SEP]"] for x in tokens]
        tokens = [self.tokenizer.convert_tokens_to_ids(x) for x in tokens]
        tokens = [x + [0] * (self._max_len - len(x)) for x in tokens]
        input_mask = [[min(1, x) for x in y] for y in tokens]
        return tokens, input_mask

    def fit(
        self,
        text,
        labels,
        max_len=512,
        device="gpu",
        use_multiple_gpus=True,
        num_epochs=1,
        batch_size=32,
        verbose=True,
    ):
        if max_len > BERTSequenceClassifier.BERT_MAX_LEN:
            print(
                "setting max_len to max allowed tokens: {}".format(
                    BERTSequenceClassifier.BERT_MAX_LEN
                )
            )
            max_len = BERTSequenceClassifier.BERT_MAX_LEN

        self._max_len = max_len

        # tokenize, truncate, and pad
        tokens, input_mask = self._get_tokens(text=text)

        device = get_device(device)
        self.model.to(device)

        # define loss function
        loss_func = nn.CrossEntropyLoss().to(device)

        # define optimizer and model parameters
        param_optimizer = list(self.model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in param_optimizer
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.01,
            },
            {
                "params": [
                    p
                    for n, p in param_optimizer
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        opt = BertAdam(optimizer_grouped_parameters, lr=2e-5)

        if use_multiple_gpus:
            if torch.cuda.device_count() > 1:
                self.model = nn.DataParallel(self.model)

        # train
        self.model.train()  # training mode
        num_examples = len(tokens)
        num_batches = int(num_examples / batch_size)
        for epoch in range(num_epochs):
            for i in range(num_batches):
                # get random batch
                start = int(random.random() * num_examples)
                end = start + batch_size
                x_batch = torch.tensor(
                    tokens[start:end], dtype=torch.long, device=device
                )
                y_batch = torch.tensor(
                    labels[start:end], dtype=torch.long, device=device
                )
                mask_batch = torch.tensor(
                    input_mask[start:end], dtype=torch.long, device=device
                )

                opt.zero_grad()
                y_h = self.model(
                    input_ids=x_batch,
                    token_type_ids=None,
                    attention_mask=mask_batch,
                    labels=None,
                )
                loss = loss_func(y_h, y_batch)
                loss.backward()
                opt.step()

                if verbose:
                    if i % (5 * num_batches) == 0:
                        print(
                            "epoch:{}/{}; batch:{}/{}; loss:{}".format(
                                epoch + 1,
                                num_epochs,
                                i + 1,
                                num_batches,
                                loss.data,
                            )
                        )
        self._is_trained = True

    def predict(self, text, device="gpu", batch_size=32):

        if not self._is_trained:
            raise Exception("Please train model before scoring")

        # tokenize, truncate, and pad
        tokens, input_mask = self._get_tokens(text=text)

        device = get_device(device)
        self.model.to(device)

        # score
        self.model.eval()
        preds = []
        for i in range(0, len(tokens), batch_size):
            x_batch = tokens[i : i + batch_size]
            mask_batch = input_mask[i : i + batch_size]
            x_batch = torch.tensor(x_batch, dtype=torch.long, device=device)
            mask_batch = torch.tensor(
                mask_batch, dtype=torch.long, device=device
            )
            with torch.no_grad():
                p_batch = self.model(
                    input_ids=x_batch,
                    token_type_ids=None,
                    attention_mask=mask_batch,
                    labels=None,
                )
            preds.append(p_batch.cpu().data.numpy())
        preds = [x.argmax(1) for x in preds]
        preds = np.concatenate(preds)
        return preds
