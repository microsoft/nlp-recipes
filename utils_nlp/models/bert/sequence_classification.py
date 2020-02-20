# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.


from collections import namedtuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from pytorch_pretrained_bert.modeling import BertForSequenceClassification
from pytorch_pretrained_bert.optimization import BertAdam
from tqdm import tqdm

from utils_nlp.models.bert.common import Language
from utils_nlp.common.pytorch_utils import (
    get_device,
    parallelize_model,
    move_model_to_device,
)

from cached_property import cached_property


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
            language, cache_dir=cache_dir, num_labels=num_labels
        )
        self.has_cuda = self.cuda

    @cached_property
    def cuda(self):
        """ cache the output of torch.cuda.is_available() """

        self.has_cuda = torch.cuda.is_available()
        return self.has_cuda

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

        device, num_gpus = get_device(num_gpus)

        self.model = move_model_to_device(self.model, device)
        self.model = parallelize_model(self.model, device, num_gpus=num_gpus)

        token_ids_tensor = torch.tensor(token_ids, dtype=torch.long)
        input_mask_tensor = torch.tensor(input_mask, dtype=torch.long)
        labels_tensor = torch.tensor(labels, dtype=torch.long)

        if token_type_ids:
            token_type_ids_tensor = torch.tensor(token_type_ids, dtype=torch.long)
            train_dataset = TensorDataset(
                token_ids_tensor,
                input_mask_tensor,
                token_type_ids_tensor,
                labels_tensor,
            )
        else:
            train_dataset = TensorDataset(
                token_ids_tensor, input_mask_tensor, labels_tensor
            )
        train_sampler = RandomSampler(train_dataset)

        train_dataloader = DataLoader(
            train_dataset, sampler=train_sampler, batch_size=batch_size
        )
        # define optimizer and model parameters
        param_optimizer = list(self.model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.01,
            },
            {
                "params": [
                    p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        num_batches = len(train_dataloader)
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

        for epoch in range(num_epochs):
            training_loss = 0
            for i, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                if token_type_ids:
                    x_batch, mask_batch, token_type_ids_batch, y_batch = tuple(
                        t.to(device) for t in batch
                    )
                else:
                    token_type_ids_batch = None
                    x_batch, mask_batch, y_batch = tuple(t.to(device) for t in batch)

                opt.zero_grad()

                y_h = self.model(
                    input_ids=x_batch,
                    token_type_ids=token_type_ids_batch,
                    attention_mask=mask_batch,
                    labels=None,
                )
                loss = loss_func(y_h, y_batch).mean()

                training_loss += loss.item()

                loss.backward()
                opt.step()
                if verbose:
                    if i % ((num_batches // 10) + 1) == 0:
                        print(
                            "epoch:{}/{}; batch:{}->{}/{}; avg loss:{:.6f}".format(
                                epoch + 1,
                                num_epochs,
                                i + 1,
                                min(i + 1 + num_batches // 10, num_batches),
                                num_batches,
                                training_loss / (i + 1),
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
        probabilities=False,
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
            probabilities (bool, optional):
                If True, the predicted probability distribution
                is also returned. Defaults to False.
        Returns:
            1darray, namedtuple(1darray, ndarray): Predicted classes or
                (classes, probabilities) if probabilities is True.
        """
        device, num_gpus = get_device(num_gpus)
        self.model = move_model_to_device(self.model, device)
        self.model = parallelize_model(self.model, device, num_gpus)

        # score
        self.model.eval()

        token_ids_tensor = torch.tensor(token_ids, dtype=torch.long)
        input_mask_tensor = torch.tensor(input_mask, dtype=torch.long)

        if token_type_ids:
            token_type_ids_tensor = torch.tensor(token_type_ids, dtype=torch.long)
            test_dataset = TensorDataset(
                token_ids_tensor, input_mask_tensor, token_type_ids_tensor
            )
        else:
            test_dataset = TensorDataset(token_ids_tensor, input_mask_tensor)

        test_sampler = SequentialSampler(test_dataset)
        test_dataloader = DataLoader(
            test_dataset, sampler=test_sampler, batch_size=batch_size
        )

        preds = []
        for i, batch in enumerate(tqdm(test_dataloader, desc="Iteration")):
            if token_type_ids:
                x_batch, mask_batch, token_type_ids_batch = tuple(
                    t.to(device) for t in batch
                )
            else:
                token_type_ids_batch = None
                x_batch, mask_batch = tuple(t.to(device) for t in batch)

            with torch.no_grad():
                p_batch = self.model(
                    input_ids=x_batch,
                    token_type_ids=token_type_ids_batch,
                    attention_mask=mask_batch,
                    labels=None,
                )
            preds.append(p_batch.cpu())

        preds = np.concatenate(preds)

        if probabilities:
            return namedtuple("Predictions", "classes probabilities")(
                preds.argmax(axis=1), nn.Softmax(dim=1)(torch.Tensor(preds)).numpy()
            )
        else:
            return preds.argmax(axis=1)
