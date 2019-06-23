# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# This script reuses some code from
# https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/examples/run_classifier.py

import random

import numpy as np
import torch
import torch.nn as nn
from pytorch_pretrained_bert.modeling import BertForSequenceClassification
from pytorch_pretrained_bert.optimization import BertAdam
from tqdm import tqdm

from utils_nlp.bert.common import Language
from utils_nlp.pytorch.device_utils import get_device, move_to_device


class BERTSequenceClassifier:
    """BERT-based sequence classifier"""

    def __init__(self, language=Language.ENGLISH, num_labels=2, cache_dir="."):
        """Initializes the classifier and the underlying pretrained model.
        Args:
            language (Language, optional): The pretrained model's language.
                                           Defaults to Language.ENGLISH.
            num_labels (int, optional): The number of unique labels in the
                training data. Defaults to 2.
            cache_dir (str, optional): Location of BERT's cache directory.
                Defaults to ".".
        """
        if num_labels < 2:
            raise ValueError("Number of labels should be at least 2.")

        self.language = language
        self.num_labels = num_labels
        self.cache_dir = cache_dir

        # create classifier
        self.model = BertForSequenceClassification.from_pretrained(
            language.value, cache_dir=cache_dir, num_labels=num_labels
        )

    def fit(
        self,
        token_ids,
        input_mask,
        labels,
        token_type_ids=None,
        num_gpus=None,
        num_epochs=1,
        batch_size=32,
        lr=2e-5,
        warmup_proportion=None,
        verbose=True,
    ):
        """Fine-tunes the BERT classifier using the given training data.
        Args:
            token_ids (list): List of training token id lists.
            input_mask (list): List of input mask lists.
            labels (list): List of training labels.
            token_type_ids (list, optional): List of lists. Each sublist
                contains segment ids indicating if the token belongs to
                the first sentence(0) or second sentence(1). Only needed
                for two-sentence tasks.
            num_gpus (int, optional): The number of gpus to use.
                                      If None is specified, all available GPUs
                                      will be used. Defaults to None.
            num_epochs (int, optional): Number of training epochs.
                Defaults to 1.
            batch_size (int, optional): Training batch size. Defaults to 32.
            lr (float): Learning rate of the Adam optimizer. Defaults to 2e-5.
            warmup_proportion (float, optional): Proportion of training to
                perform linear learning rate warmup for. E.g., 0.1 = 10% of
                training. Defaults to None.
            verbose (bool, optional): If True, shows the training progress and
                loss values. Defaults to True.
        """

        device = get_device("cpu" if num_gpus == 0 else "gpu")
        self.model = move_to_device(self.model, device, num_gpus)

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
                ]
            },
        ]

        num_examples = len(token_ids)
        num_batches = int(num_examples / batch_size)
        num_train_optimization_steps = num_batches * num_epochs

        if warmup_proportion is None:
            opt = BertAdam(optimizer_grouped_parameters, lr=lr)
        else:
            opt = BertAdam(
                optimizer_grouped_parameters,
                lr=lr,
                t_total=num_train_optimization_steps,
                warmup=warmup_proportion,
            )

        # define loss function
        loss_func = nn.CrossEntropyLoss().to(device)

        # train
        self.model.train()  # training mode

        token_type_ids_batch = None
        for epoch in range(num_epochs):
            for i in range(num_batches):

                # get random batch
                start = int(random.random() * num_examples)
                end = start + batch_size
                x_batch = torch.tensor(
                    token_ids[start:end], dtype=torch.long, device=device
                )
                y_batch = torch.tensor(
                    labels[start:end], dtype=torch.long, device=device
                )
                mask_batch = torch.tensor(
                    input_mask[start:end], dtype=torch.long, device=device
                )

                if token_type_ids is not None:
                    token_type_ids_batch = torch.tensor(
                        token_type_ids[start:end],
                        dtype=torch.long,
                        device=device,
                    )

                opt.zero_grad()

                y_h = self.model(
                    input_ids=x_batch,
                    token_type_ids=token_type_ids_batch,
                    attention_mask=mask_batch,
                    labels=None,
                )

                loss = loss_func(y_h, y_batch).mean()
                loss.backward()
                opt.step()
                if verbose:
                    if i % ((num_batches // 10) + 1) == 0:
                        print(
                            "epoch:{}/{}; batch:{}->{}/{}; loss:{:.6f}".format(
                                epoch + 1,
                                num_epochs,
                                i + 1,
                                min(i + 1 + num_batches // 10, num_batches),
                                num_batches,
                                loss.data,
                            )
                        )
        # empty cache
        del [x_batch, y_batch, mask_batch, token_type_ids_batch]
        torch.cuda.empty_cache()

    def predict(
        self,
        token_ids,
        input_mask,
        token_type_ids=None,
        num_gpus=None,
        batch_size=32,
    ):
        """Scores the given dataset and returns the predicted classes.
        Args:
            token_ids (list): List of training token lists.
            input_mask (list): List of input mask lists.
            token_type_ids (list, optional): List of lists. Each sublist
                contains segment ids indicating if the token belongs to
                the first sentence(0) or second sentence(1). Only needed
                for two-sentence tasks.
            num_gpus (int, optional): The number of gpus to use.
                                      If None is specified, all available GPUs
                                      will be used. Defaults to None.
            batch_size (int, optional): Scoring batch size. Defaults to 32.
        Returns:
            [ndarray]: Predicted classes.
        """

        device = get_device("cpu" if num_gpus == 0 else "gpu")
        self.model = move_to_device(self.model, device, num_gpus)

        # score
        self.model.eval()
        preds = []
        with tqdm(total=len(token_ids)) as pbar:
            for i in range(0, len(token_ids), batch_size):
                x_batch = token_ids[i : i + batch_size]
                mask_batch = input_mask[i : i + batch_size]
                x_batch = torch.tensor(
                    x_batch, dtype=torch.long, device=device
                )
                mask_batch = torch.tensor(
                    mask_batch, dtype=torch.long, device=device
                )
                token_type_ids_batch = None
                if token_type_ids is not None:
                    token_type_ids_batch = torch.tensor(
                        token_type_ids[i : i + batch_size],
                        dtype=torch.long,
                        device=device,
                    )
                with torch.no_grad():
                    p_batch = self.model(
                        input_ids=x_batch,
                        token_type_ids=token_type_ids_batch,
                        attention_mask=mask_batch,
                        labels=None,
                    )
                preds.append(p_batch.cpu().data.numpy())
                if i % batch_size == 0:
                    pbar.update(batch_size)
        preds = [x.argmax(1) for x in preds]
        preds = np.concatenate(preds)
        return preds
