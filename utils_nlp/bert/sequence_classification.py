# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import random
import numpy as np
import torch
import torch.nn as nn
from pytorch_pretrained_bert.modeling import BertForSequenceClassification
from pytorch_pretrained_bert.optimization import BertAdam
from tqdm import tqdm
from utils_nlp.bert.common import BERT_MAX_LEN, Language
from utils_nlp.pytorch.device_utils import get_device, move_to_device


class SequenceClassifier:
    """BERT-based sequence classifier"""

    def __init__(self, language=Language.ENGLISH, num_labels=2, cache_dir="."):
        """Initializes the classifier and the underlying pretrained model.
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
        verbose=True,
    ):
        """Fine-tunes the BERT classifier using the given training data.
        Args:
            token_ids (list): List of training token id lists.
            input_mask (list): List of input mask lists.
            labels (list): List of training labels.
            device (str, optional): Device used for training ("cpu" or "gpu").
                                    Defaults to "gpu".
            num_gpus (int, optional): The number of gpus to use.
                                      If None is specified, all available GPUs will be used.
                                      Defaults to None.
            num_epochs (int, optional): Number of training epochs. Defaults to 1.
            batch_size (int, optional): Training batch size. Defaults to 32.
            lr (float): Learning rate of the Adam optimizer. Defaults to 2e-5.
            verbose (bool, optional): If True, shows the training progress and loss values.
                                      Defaults to True.
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

        opt = BertAdam(optimizer_grouped_parameters, lr=lr)

        # define loss function
        # loss_func = nn.CrossEntropyLoss().to(device)

        class WeightedLogLoss(torch.nn.Module):
            def __init__(self):
                super(WeightedLogLoss, self).__init__()

            def forward(self, outputs, labels):
                logits = torch.nn.functional.log_softmax(outputs, dim=-1)
                # labels = torch.transpose(labels, 0, 1)

                weighted_logits = labels.float() * logits

                # print("labels:   {}".format(labels.size()))
                # print("logits:   {}".format(logits.size()))
                # print("weighted: {}".format(weighted_logits.size()))

                return -torch.sum(weighted_logits)

        loss_func = WeightedLogLoss().to(device)

        # train
        self.model.train()  # training mode
        num_examples = len(token_ids)
        num_batches = int(num_examples / batch_size)

        x_batch = y_batch = mask_batch = token_type_ids_batch = None

        with tqdm(total=num_epochs * num_examples) as pbar:
            for epoch in range(num_epochs):
                for i in range(num_batches):
                    # data is assumed to be shuffled
                    start = i * batch_size
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
                            token_type_ids[start:end], dtype=torch.long, device=device
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

                    pbar.update(batch_size)
                    if i % 10 == 0:
                        pbar.set_description(
                            "epoch:{}/{}; loss:{:.6f}".format(epoch, num_epochs, loss.data))

        # empty cache
        del [x_batch, y_batch, mask_batch, token_type_ids_batch]
        torch.cuda.empty_cache()

    def predict(self, token_ids, input_mask, token_type_ids=None, num_gpus=1, batch_size=32):
        """Scores the given dataset and returns the predicted classes.
        Args:
            token_ids (list): List of training token lists.
            input_mask (list): List of input mask lists.
            num_gpus (int, optional): The number of gpus to use.
                                      If None is specified, all available GPUs will be used.
                                      Defaults to 1.
            batch_size (int, optional): Scoring batch size. Defaults to 32.
        Returns:
            [ndarray]: Predicted classes.
        """

        device = get_device("cpu" if num_gpus == 0 else "gpu")

        # print(self.model)

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
                        token_type_ids[i : i +
                                       batch_size], dtype=torch.long, device=device
                    )

                with torch.no_grad():
                    p_batch = self.model(
                        input_ids=x_batch,
                        token_type_ids=token_type_ids_batch,
                        attention_mask=mask_batch,
                        labels=None,
                    )

                    # final softmax
                    p_batch = torch.nn.functional.softmax(p_batch, -1)

                # only take prob for 1-class
                preds.append(p_batch.cpu().data.numpy()[:, 0])
                if i % batch_size == 0:
                    pbar.update(batch_size)

        preds = np.concatenate(preds)
        return preds
