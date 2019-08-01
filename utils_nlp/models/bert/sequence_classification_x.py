# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import logging

import horovod.torch as hvd
import numpy as np
import torch.nn as nn
from torch.utils.data import TensorDataset
import torch.utils.data.distributed
from pytorch_pretrained_bert.modeling import BertForSequenceClassification
from pytorch_pretrained_bert.optimization import BertAdam
from tqdm import tqdm

from utils_nlp.models.bert.common import Language

from utils_nlp.models.bert.common import get_dataset_multiple_files
from utils_nlp.common.pytorch_utils import get_device, move_to_device

logger = logging.getLogger(__name__)
hvd.init()
torch.manual_seed(42)

if torch.cuda.is_available():
    # Horovod: pin GPU to local rank.
    torch.cuda.set_device(hvd.local_rank())
    torch.cuda.manual_seed(42)


class BERTSequenceDistClassifier:
    """Distributed BERT-based sequence classifier"""

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
        self.kwargs = (
            {"num_workers": 1, "pin_memory": True}
            if torch.cuda.is_available()
            else {}
        )

        # create classifier
        self.model = BertForSequenceClassification.from_pretrained(
            language.value, num_labels=num_labels
        )

    def fit(
        self,
        token_ids,
        input_mask,
        labels,
        token_type_ids=None,
        input_files=None,
        num_gpus=None,
        num_epochs=1,
        batch_size=32,
        lr=2e-5,
        warmup_proportion=None,
        verbose=True,
        fp16_allreduce=False,
    ):
        """fine-tunes the bert classifier using the given training data.

        args:
            input_files(list, required): list of paths to the training data files.
            token_ids (list): List of training token id lists.
            input_mask (list): List of input mask lists.
            labels (list): List of training labels.
            token_type_ids (list, optional): List of lists. Each sublist
                contains segment ids indicating if the token belongs to
                the first sentence(0) or second sentence(1). Only needed
                for two-sentence tasks.
            num_gpus (int, optional): the number of gpus to use.
                                      if none is specified, all available gpus
                                      will be used. defaults to none.
            num_epochs (int, optional): number of training epochs.
                defaults to 1.
            batch_size (int, optional): training batch size. defaults to 32.
            lr (float): learning rate of the adam optimizer. defaults to 2e-5.
            warmup_proportion (float, optional): proportion of training to
                perform linear learning rate warmup for. e.g., 0.1 = 10% of
                training. defaults to none.
            verbose (bool, optional): if true, shows the training progress and
                loss values. defaults to true.
            fp16_allreduce(bool, optional)L if true, use fp16 compression during allreduce
        """

        if input_files is not None:
            train_dataset = get_dataset_multiple_files(input_files)
        else:
            token_ids_tensor = torch.tensor(token_ids, dtype=torch.long)
            input_mask_tensor = torch.tensor(input_mask, dtype=torch.long)
            labels_tensor = torch.tensor(labels, dtype=torch.long)

            if token_type_ids:
                token_type_ids_tensor = torch.tensor(
                    token_type_ids, dtype=torch.long
                )
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

        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=hvd.size(), rank=hvd.rank()
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            **self.kwargs
        )

        device = get_device("cpu" if num_gpus == 0 else "gpu")
        self.model.cuda()

        hvd.broadcast_parameters(self.model.state_dict(), root_rank=0)
        # hvd.broadcast_optimizer_state(optimizer, root_rank=0)

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
                ]
            },
        ]

        num_examples = len(train_dataset)
        num_batches = int(num_examples / batch_size)
        num_train_optimization_steps = num_batches * num_epochs

        if warmup_proportion is None:
            optimizer = BertAdam(
                optimizer_grouped_parameters, lr=lr * hvd.size()
            )
        else:
            optimizer = BertAdam(
                optimizer_grouped_parameters,
                lr=lr * hvd.size(),
                t_total=num_train_optimization_steps,
                warmup=warmup_proportion,
            )

        # Horovod: (optional) compression algorithm.
        compression = (
            hvd.Compression.fp16 if fp16_allreduce else hvd.Compression.none
        )

        # Horovod: wrap optimizer with DistributedOptimizer.
        optimizer = hvd.DistributedOptimizer(
            optimizer,
            named_parameters=self.model.named_parameters(),
            compression=compression,
        )

        # Horovod: set epoch to sampler for shuffling.
        for epoch in range(num_epochs):
            self.model.train()
            train_sampler.set_epoch(epoch)
            for batch_idx, batch in enumerate(train_loader):

                if token_type_ids:
                    x_batch, mask_batch, token_type_ids_batch, y_batch = tuple(
                        t.to(device) for t in batch
                    )
                else:
                    token_type_ids_batch = None
                    x_batch, mask_batch, y_batch = tuple(
                        t.to(device) for t in batch
                    )

                optimizer.zero_grad()

                output = self.model(
                    input_ids=x_batch, attention_mask=mask_batch, labels=None
                )

                loss = loss_func(output, y_batch).mean()
                loss.backward()
                optimizer.step()
                if verbose and (batch_idx % ((num_batches // 10) + 1)) == 0:
                    # Horovod: use train_sampler to determine the number of examples in
                    # this worker's partition.
                    print(
                        "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                            epoch,
                            batch_idx * len(x_batch),
                            len(train_sampler),
                            100.0 * batch_idx / len(train_loader),
                            loss.item(),
                        )
                    )

        # empty cache
        torch.cuda.empty_cache()

    def predict(
        self,
        token_ids,
        input_mask,
        token_type_ids=None,
        input_files=None,
        num_gpus=None,
        batch_size=32,
        probabilities=False,
    ):
        """Scores the given set of train files and returns the predicted classes.

        Args:
            input_files(list, required): list of paths to the test data files.
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
            1darray, dict(1darray, 1darray, ndarray): Predicted classes and target labels or
                a dictionary with classes, target labels, probabilities) if probabilities is True.
        """

        if input_files is not None:
            test_dataset = get_dataset_multiple_files(input_files)

        else:
            token_ids_tensor = torch.tensor(token_ids, dtype=torch.long)
            input_mask_tensor = torch.tensor(input_mask, dtype=torch.long)

            if token_type_ids:
                token_type_ids_tensor = torch.tensor(
                    token_type_ids, dtype=torch.long
                )
                test_dataset = TensorDataset(
                    token_ids_tensor, input_mask_tensor, token_type_ids_tensor
                )
            else:
                test_dataset = TensorDataset(token_ids_tensor, input_mask_tensor)

        # Horovod: use DistributedSampler to partition the test data.
        test_sampler = torch.utils.data.sampler.SequentialSampler(test_dataset)

        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            sampler=test_sampler,
            **self.kwargs
        )

        device = get_device()
        self.model = move_to_device(self.model, device, num_gpus)
        self.model.eval()
        preds = []
        labels_test = []

        with tqdm(total=len(test_loader)) as pbar:
            for i, (tokens, mask, target) in enumerate(test_loader):
                if torch.cuda.is_available():
                    tokens, mask, target = (
                        tokens.cuda(),
                        mask.cuda(),
                        target.cuda(),
                    )

                with torch.no_grad():
                    p_batch = self.model(
                        input_ids=tokens, attention_mask=mask, labels=None
                    )
                preds.append(p_batch.cpu())
                labels_test.append(target.cpu())
                if i % batch_size == 0:
                    pbar.update(batch_size)

        preds = np.concatenate(preds)
        labels_test = np.concatenate(labels_test)

        if probabilities:
            return {
                "Predictions": preds.argmax(axis=1),
                "Target": labels_test,
                "classes probabilities": nn.Softmax(dim=1)(
                    torch.Tensor(preds)
                ).numpy(),
            }
        else:
            return preds.argmax(axis=1), labels_test
