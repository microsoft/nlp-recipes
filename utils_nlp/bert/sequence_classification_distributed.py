# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# This script reuses some code from
# https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/examples/run_classifier.py

from collections import namedtuple

import os
import numpy as np
import torch
import torch.nn as nn
from pytorch_pretrained_bert.modeling import BertForSequenceClassification
from tqdm import tqdm
from utils_nlp.bert.common import Language
from utils_nlp.pytorch.device_utils import get_device, move_to_device


class BERTSequenceClassifier:
    """BERT-based sequence classifier"""

    def __init__(self, language=Language.ENGLISH, num_labels=2, cache_dir="."):
        if num_labels < 2:
            raise ValueError("Number of labels should be at least 2.")

        self.language = language
        self.num_labels = num_labels
        self.cache_dir = cache_dir

        # create classifier
        print("=============== create bert model ==========================")
        self.model = BertForSequenceClassification.from_pretrained(
            language.value, cache_dir=cache_dir, num_labels=num_labels
        )


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
        self.optimizer_params = optimizer_grouped_parameters
        self.name_parameters = self.model.named_parameters()
        self.state_dict = self.model.state_dict()

    def fit(
        self,
        train_loader,
        bert_optimizer=None,
        num_epochs=1,
        num_gpus=0,
        rank = 0):

        # define loss function
        device = get_device("cpu" if num_gpus == 0 else "gpu")
        loss_func = nn.CrossEntropyLoss().to(device)

        # train
        self.model.train()  # training mode

        token_type_ids_batch = None

        num_print = 100
        for epoch in range(1, num_epochs + 1):
            for batch_idx, data in enumerate(train_loader):
                x_batch = data['token_ids']
                y_batch = data['labels']
                mask_batch = data['input_mask']

                if data['token_type_ids'] is not None:
                    token_type_ids_batch = data['token_type_ids']

                bert_optimizer.zero_grad()

                y_h = self.model(input_ids=x_batch,
                            token_type_ids=token_type_ids_batch,
                            attention_mask=mask_batch,
                            labels=None,)

                # not sure of this part
                loss = loss_func(y_h, y_batch).mean()
                loss.backward()

                bert_optimizer.synchronize()
                bert_optimizer.step()

                if batch_idx % num_print == 0:
                    print(
                        "epoch:{}/{}; batch:{}; loss:{:.6f}".format(
                            epoch,
                            num_epochs,
                            batch_idx + 1,
                            loss.data,
                        )
                )

        # Save the model to the outputs directory for capture
        output_dir = 'outputs'
        weights_name = 'bert-large-uncased'
        os.makedirs(output_dir, exist_ok=True)


        if rank == 0:
            # Save a trained model, configuration and tokenizer
            model_to_save = self.model.module if hasattr(self.model, 'module') else self.model  # Only save the model it-self

            # If we save using the predefined names, we can load using `from_pretrained`
            output_model_file = 'outputs/bert-large-uncased'
            output_config_file = 'outputs/bert_config.json'

            torch.save(model_to_save.state_dict(), output_model_file)
            model_to_save.config.to_json_file(output_config_file)

        else:
            self.model.to(device)

        del [x_batch, y_batch, mask_batch, token_type_ids_batch]
        torch.cuda.empty_cache()

    def predict(
        self,
        test_loader,
        num_gpus=None,
        batch_size=32,
        probabilities=False,
    ):

        #device = get_device("cpu" if num_gpus == 0 else "gpu")
        #self.model = move_to_device(self.model, device, num_gpus)

        # score
        self.model.eval()

        preds = []
        with tqdm(total=batch_size) as pbar:
            for i, data in enumerate(test_loader):
                x_batch = data['token_ids']
                mask_batch = data['input_mask']

                token_type_ids_batch = None
                if data['token_type_ids'] is not None:
                    token_type_ids_batch = data['token_type_ids']

                with torch.no_grad():
                    p_batch = self.model(
                        input_ids=x_batch,
                        token_type_ids=token_type_ids_batch,
                        attention_mask=mask_batch,
                        labels=None,
                    )
                preds.append(p_batch.cpu())
                if i % batch_size == 0:
                    pbar.update(batch_size)

        preds = np.concatenate(preds)

        if probabilities:
            return namedtuple("Predictions", "classes probabilities")(
                preds.argmax(axis=1),
                nn.Softmax(dim=1)(torch.Tensor(preds)).numpy(),
            )
        else:
            return preds.argmax(axis=1)
