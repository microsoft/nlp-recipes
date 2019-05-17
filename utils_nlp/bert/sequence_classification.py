# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import random
import numpy as np
import torch.nn as nn
import torch
from pytorch_pretrained_bert.modeling import BertForSequenceClassification
from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.tokenization import BertTokenizer
from utils_nlp.pytorch.device import get_device
from utils_nlp.bert.common import Language


class BERTSequenceClassifier:
    """BERT-based sequence classifier"""

    BERT_MAX_LEN = 512

    def __init__(self, language=Language.ENGLISH, num_labels=2, cache_dir="."):
        """Initializes the classifier and the underlying pretrained model and tokenizer.
        Args:
            language (Language, optional): The pretrained model's language.
                                           Defaults to Language.ENGLISH.
            num_labels (int, optional): The number of unique labels in the training data.
                                        Defaults to 2.
            cache_dir (str, optional): Location of BERT's cache directory. Defaults to ".".
        """
        if num_labels < 2:
            raise Exception("Number of labels should be at least 2.")

        self.language = language
        self.num_labels = num_labels
        self._max_len = BERTSequenceClassifier.BERT_MAX_LEN
        self.cache_dir = cache_dir
        # create tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(
            language.value, do_lower_case=False, cache_dir=cache_dir
        )
        # create classifier
        self.model = BertForSequenceClassification.from_pretrained(
            language.value, cache_dir=cache_dir, num_labels=num_labels
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

    def get_model(self):
        """Returns the underlying PyTorch model
             BertForSequenceClassification or DataParallel (when multiple GPUs are used)."""
        return self.model

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
        """Fine-tunes the BERT classifier using the given training data.
        Args:
            text (list): List of training text documents.
            labels (list): List of training labels.
            max_len (int, optional): Maximum number of tokens
                                     (documents will be truncated or padded).
                                     Defaults to 512.
            device (str, optional): Device used for training ("cpu" or "gpu").
                                    Defaults to "gpu".
            use_multiple_gpus (bool, optional): Whether multiple GPUs will be used for training.
                                                Defaults to True.
            num_epochs (int, optional): Number of training epochs. Defaults to 1.
            batch_size (int, optional): Training batch size. Defaults to 32.
            verbose (bool, optional): If True, shows the training progress and loss values.
                                      Defaults to True.
        """
        device = get_device(device)
        self.model.to(device)

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
                    if i % (2 * batch_size) == 0:
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
        """Scores the given dataset and returns the predicted classes.
        Args:
            text (list): List of text documents to score.
            device (str, optional): Device used for scoring ("cpu" or "gpu"). Defaults to "gpu".
            batch_size (int, optional): Scoring batch size. Defaults to 32.
        Returns:
            [ndarray]: Predicted classes.
        """

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
