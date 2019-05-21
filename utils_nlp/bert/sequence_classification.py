# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import random
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch
from pytorch_pretrained_bert.modeling import BertForSequenceClassification
from pytorch_pretrained_bert.optimization import BertAdam
from utils_nlp.pytorch.device import get_device
from utils_nlp.bert.common import Language, BERT_MAX_LEN


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
        self._model = BertForSequenceClassification.from_pretrained(
            language.value, cache_dir=cache_dir, num_labels=num_labels
        )
        

    def get_model(self):
        """Returns the underlying PyTorch model
             BertForSequenceClassification or DataParallel (when multiple GPUs are used)."""
        return self._model


    def fit(
        self,
        tokens,
        input_mask,
        labels,        
        device="gpu",
        use_multiple_gpus=True,
        num_epochs=1,
        batch_size=32,
        verbose=True,
    ):
        """Fine-tunes the BERT classifier using the given training data.
        Args:
            tokens (list): List of training token lists.
            input_mask (list): List of input mask lists.
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
        self._model.to(device)

        # define loss function
        loss_func = nn.CrossEntropyLoss().to(device)

        # define optimizer and model parameters
        param_optimizer = list(self._model.named_parameters())
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

        n_gpus = 0
        if use_multiple_gpus:
            n_gpus = torch.cuda.device_count()
            if n_gpus > 1:
                if isinstance(self._model, nn.DataParallel):
                    self._model = nn.DataParallel(self._model)

        # train
        self._model.train()  # training mode
        num_examples = len(tokens)
        num_batches = int(num_examples / batch_size)

        with tqdm(total=num_epochs * num_batches) as pbar:
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
                    y_h = self._model(
                        input_ids=x_batch,
                        token_type_ids=None,
                        attention_mask=mask_batch,
                        labels=None,
                    )

                    loss = loss_func(y_h, y_batch)
                    if n_gpus > 1:
                        loss = loss.mean()

                    loss.backward()
                    opt.step(tqdm)
                    pbar.update()
                    if verbose:
                        if i % (batch_size) == 0:
                            print(
                                "epoch:{}/{}; batch:{}/{}; loss:{}".format(
                                    epoch + 1,
                                    num_epochs,
                                    i + 1,
                                    num_batches,
                                    loss.data,
                                )
                            )

    def predict(self, tokens, input_mask, device="gpu", batch_size=32):
        """Scores the given dataset and returns the predicted classes.
        Args:
            tokens (list): List of training token lists.
            input_mask (list): List of input mask lists.
            device (str, optional): Device used for scoring ("cpu" or "gpu"). Defaults to "gpu".
            batch_size (int, optional): Scoring batch size. Defaults to 32.
        Returns:
            [ndarray]: Predicted classes.
        """

        device = get_device(device)
        self._model.to(device)

        # score
        self._model.eval()
        preds = []
        for i in tqdm(range(0, len(tokens), batch_size)):
            x_batch = tokens[i : i + batch_size]
            mask_batch = input_mask[i : i + batch_size]
            x_batch = torch.tensor(x_batch, dtype=torch.long, device=device)
            mask_batch = torch.tensor(
                mask_batch, dtype=torch.long, device=device
            )
            with torch.no_grad():
                p_batch = self._model(
                    input_ids=x_batch,
                    token_type_ids=None,
                    attention_mask=mask_batch,
                    labels=None,
                )
            preds.append(p_batch.cpu().data.numpy())
        preds = [x.argmax(1) for x in preds]
        preds = np.concatenate(preds)
        return preds
