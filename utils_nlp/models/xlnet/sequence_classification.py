# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Utilities for Xlnet Sequence Classification"""
import numpy as np
from collections import namedtuple
import torch
import torch.nn as nn
from transformers import (
    XLNetConfig,
    XLNetForSequenceClassification,
    AdamW,
    WarmupLinearSchedule,
)
from tqdm import tqdm
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
from utils_nlp.common.pytorch_utils import get_device, move_to_device
from utils_nlp.models.xlnet.common import Language
import mlflow
import mlflow.pytorch
import os


class XLNetSequenceClassifier:
    """XLNet-based sequence classifier"""

    def __init__(
        self,
        language=Language.ENGLISHCASED,
        num_labels=5,
        cache_dir=".",
        num_gpus=None,
        num_epochs=1,
        batch_size=8,
        lr=5e-5,
        adam_eps=1e-8,
        warmup_steps=0,
        weight_decay=0.0,
        max_grad_norm=1.0,
    ):
        """Initializes the classifier and the underlying pretrained model.

        Args:
            language (Language, optional): The pretrained model's language.
                                           Defaults to 'xlnet-base-cased'.
            num_labels (int, optional): The number of unique labels in the
                training data. Defaults to 5.
            cache_dir (str, optional): Location of XLNet's cache directory.
                Defaults to ".".
            num_gpus (int, optional): The number of gpus to use.
                                      If None is specified, all available GPUs
                                      will be used. Defaults to None.
            num_epochs (int, optional): Number of training epochs.
                Defaults to 1.
            batch_size (int, optional): Training batch size. Defaults to 8.
            lr (float): Learning rate of the Adam optimizer. Defaults to 5e-5.
            adam_eps (float, optional): term added to the denominator to improve
                                        numerical stability. Defaults to 1e-8.
            warmup_steps (int, optional): Number of steps in which to increase
                                        learning rate linearly from 0 to 1. Defaults to 0.
            weight_decay (float, optional): Weight decay. Defaults to 0.
            max_grad_norm (float, optional): Maximum norm for the gradients. Defaults to 1.0
        """

        if num_labels < 2:
            raise ValueError("Number of labels should be at least 2.")

        self.language = language
        self.num_labels = num_labels
        self.cache_dir = cache_dir

        self.num_gpus = num_gpus
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.adam_eps = adam_eps
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm

        # create classifier
        self.config = XLNetConfig.from_pretrained(
            self.language.value, num_labels=num_labels, cache_dir=cache_dir
        )
        self.model = XLNetForSequenceClassification(self.config)

    def fit(
        self,
        token_ids,
        input_mask,
        labels,
        val_token_ids,
        val_input_mask,
        val_labels,
        token_type_ids=None,
        val_token_type_ids=None,
        verbose=True,
        logging_steps=0,
        save_steps=0,
        val_steps=0,
    ):
        """Fine-tunes the XLNet classifier using the given training data.

        Args:
            token_ids (list): List of training token id lists.
            input_mask (list): List of input mask lists.
            labels (list): List of training labels.
            token_type_ids (list, optional): List of lists. Each sublist
                contains segment ids indicating if the token belongs to
                the first sentence(0) or second sentence(1). Only needed
                for two-sentence tasks.
            verbose (bool, optional): If True, shows the training progress and
                loss values. Defaults to True.
        """

        device, num_gpus = get_device(self.num_gpus)
        self.model = move_to_device(self.model, device, self.num_gpus)

        token_ids_tensor = torch.tensor(token_ids, dtype=torch.long)
        input_mask_tensor = torch.tensor(input_mask, dtype=torch.long)
        labels_tensor = torch.tensor(labels, dtype=torch.long)

        val_token_ids_tensor = torch.tensor(val_token_ids, dtype=torch.long)
        val_input_mask_tensor = torch.tensor(val_input_mask, dtype=torch.long)
        val_labels_tensor = torch.tensor(val_labels, dtype=torch.long)

        if token_type_ids:
            token_type_ids_tensor = torch.tensor(token_type_ids, dtype=torch.long)
            val_token_type_ids_tensor = torch.tensor(val_token_type_ids, dtype=torch.long)

            train_dataset = TensorDataset(
                token_ids_tensor, input_mask_tensor, token_type_ids_tensor, labels_tensor
            )

            val_dataset = TensorDataset(
                val_token_ids_tensor,
                val_input_mask_tensor,
                val_token_type_ids_tensor,
                val_labels_tensor,
            )

        else:

            train_dataset = TensorDataset(token_ids_tensor, input_mask_tensor, labels_tensor)

            val_dataset = TensorDataset(
                val_token_ids_tensor, val_input_mask_tensor, val_labels_tensor
            )

        # define optimizer and model parameters
        param_optimizer = list(self.model.named_parameters())
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        val_sampler = RandomSampler(val_dataset)

        val_dataloader = DataLoader(val_dataset, sampler=val_sampler, batch_size=self.batch_size)

        num_examples = len(token_ids)
        num_batches = int(np.ceil(num_examples / self.batch_size))
        num_train_optimization_steps = num_batches * self.num_epochs

        optimizer = AdamW(optimizer_grouped_parameters, lr=self.lr, eps=self.adam_eps)
        scheduler = WarmupLinearSchedule(
            optimizer, warmup_steps=self.warmup_steps, t_total=num_train_optimization_steps
        )

        global_step = 0
        self.model.train()
        optimizer.zero_grad()
        for epoch in range(self.num_epochs):

            train_sampler = RandomSampler(train_dataset)

            train_dataloader = DataLoader(
                train_dataset, sampler=train_sampler, batch_size=self.batch_size
            )

            tr_loss = 0.0
            logging_loss = 0.0
            val_loss = 0.0

            for i, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                if token_type_ids:
                    x_batch, mask_batch, token_type_ids_batch, y_batch = tuple(
                        t.to(device) for t in batch
                    )
                else:
                    token_type_ids_batch = None
                    x_batch, mask_batch, y_batch = tuple(t.to(device) for t in batch)

                outputs = self.model(
                    input_ids=x_batch,
                    token_type_ids=token_type_ids_batch,
                    attention_mask=mask_batch,
                    labels=y_batch,
                )

                loss = outputs[0]  # model outputs are always tuple in pytorch-transformers

                loss.sum().backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                tr_loss += loss.sum().item()
                optimizer.step()
                # Update learning rate schedule
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                # logging of learning rate and loss
                if logging_steps > 0 and global_step % logging_steps == 0:
                    mlflow.log_metric("learning rate", scheduler.get_lr()[0], step=global_step)
                    mlflow.log_metric(
                        "training loss",
                        (tr_loss - logging_loss) / (logging_steps * self.batch_size),
                        step=global_step,
                    )
                    logging_loss = tr_loss
                # model checkpointing
                if save_steps > 0 and global_step % save_steps == 0:
                    checkpoint_dir = os.path.join(os.getcwd(), "checkpoints")
                    if not os.path.isdir(checkpoint_dir):
                        os.makedirs(checkpoint_dir)
                    checkpoint_path = checkpoint_dir + "/" + str(global_step) + ".pth"
                    torch.save(self.model.state_dict(), checkpoint_path)
                    mlflow.log_artifact(checkpoint_path)
                # model validation
                if val_steps > 0 and global_step % val_steps == 0:
                    # run model on validation set
                    self.model.eval()
                    val_loss = 0.0
                    for j, val_batch in enumerate(val_dataloader):
                        if token_type_ids:
                            val_x_batch, val_mask_batch, val_token_type_ids_batch, val_y_batch = tuple(
                                t.to(device) for t in val_batch
                            )
                        else:
                            token_type_ids_batch = None
                            val_x_batch, val_mask_batch, val_y_batch = tuple(
                                t.to(device) for t in val_batch
                            )
                        val_outputs = self.model(
                            input_ids=val_x_batch,
                            token_type_ids=val_token_type_ids_batch,
                            attention_mask=val_mask_batch,
                            labels=val_y_batch,
                        )
                        vloss = val_outputs[0]
                        val_loss += vloss.sum().item()
                    mlflow.log_metric(
                        "validation loss", val_loss / len(val_dataset), step=global_step
                    )
                    self.model.train()

                if verbose:
                    if i % ((num_batches // 10) + 1) == 0:
                        if val_loss > 0:
                            print(
                                "epoch:{}/{}; batch:{}->{}/{}; average training loss:{:.6f};\
                                 average val loss:{:.6f}".format(
                                    epoch + 1,
                                    self.num_epochs,
                                    i + 1,
                                    min(i + 1 + num_batches // 10, num_batches),
                                    num_batches,
                                    tr_loss / (i + 1),
                                    val_loss / (j + 1),
                                )
                            )
                        else:
                            print(
                                "epoch:{}/{}; batch:{}->{}/{}; average train loss:{:.6f}".format(
                                    epoch + 1,
                                    self.num_epochs,
                                    i + 1,
                                    min(i + 1 + num_batches // 10, num_batches),
                                    num_batches,
                                    tr_loss / (i + 1),
                                )
                            )
        checkpoint_dir = os.path.join(os.getcwd(), "checkpoints")
        if not os.path.isdir(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        checkpoint_path = checkpoint_dir + "/" + "final" + ".pth"
        torch.save(self.model.state_dict(), checkpoint_path)
        mlflow.log_artifact(checkpoint_path)
        # empty cache
        del [x_batch, y_batch, mask_batch, token_type_ids_batch]
        if val_steps > 0:
            del [val_x_batch, val_y_batch, val_mask_batch, val_token_type_ids_batch]
        torch.cuda.empty_cache()

    def predict(
        self,
        token_ids,
        input_mask,
        token_type_ids=None,
        num_gpus=None,
        batch_size=8,
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
            batch_size (int, optional): Scoring batch size. Defaults to 8.
            probabilities (bool, optional):
                If True, the predicted probability distribution
                is also returned. Defaults to False.
        Returns:
            1darray, namedtuple(1darray, ndarray): Predicted classes or
                (classes, probabilities) if probabilities is True.
        """

        device, num_gpus = get_device(num_gpus)
        self.model = move_to_device(self.model, device, num_gpus)

        self.model.eval()
        preds = []

        with tqdm(total=len(token_ids)) as pbar:
            for i in range(0, len(token_ids), batch_size):
                start = i
                end = start + batch_size
                x_batch = torch.tensor(token_ids[start:end], dtype=torch.long, device=device)
                mask_batch = torch.tensor(input_mask[start:end], dtype=torch.long, device=device)

                token_type_ids_batch = torch.tensor(
                    token_type_ids[start:end], dtype=torch.long, device=device
                )

                with torch.no_grad():
                    pred_batch = self.model(
                        input_ids=x_batch,
                        token_type_ids=token_type_ids_batch,
                        attention_mask=mask_batch,
                        labels=None,
                    )
                    preds.append(pred_batch[0].cpu())
                    if i % batch_size == 0:
                        pbar.update(batch_size)

            preds = np.concatenate(preds)

            if probabilities:
                return namedtuple("Predictions", "classes probabilities")(
                    preds.argmax(axis=1), nn.Softmax(dim=1)(torch.Tensor(preds)).numpy()
                )
            else:
                return preds.argmax(axis=1)
