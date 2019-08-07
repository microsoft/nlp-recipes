import numpy as np
from collections import namedtuple
import torch
import torch.nn as nn
from pytorch_transformers import (WEIGHTS_NAME,XLNetConfig,XLNetForSequenceClassification)
from tqdm import tqdm
from torch.utils.data import (
    DataLoader,
    RandomSampler,
    SequentialSampler,
    TensorDataset,
)
from pytorch_transformers import AdamW, WarmupLinearSchedule
from utils_nlp.common.pytorch_utils import get_device, move_to_device
from utils_nlp.models.xlnet.common import Language
import random

class XLNetSequenceClassifier:
    """XLNet-based sequence classifier"""
    
    def __init__(self, language=Language.ENGLISHCASED, num_labels=5, cache_dir='.'):
        """Initializes the classifier and the underlying pretrained model.
        
        Args:
            language (Language, optional): The pretrained model's language.
                                           Defaults to 'xlnet-base-cased'.
            num_labels (int, optional): The number of unique labels in the
                training data. Defaults to 5.
            cache_dir (str, optional): Location of XLNet's cache directory.
                Defaults to ".".
        """
        
        if num_labels < 2:
            raise ValueError("Number of labels should be at least 2.")
        
        self.language = language
        self.num_labels = num_labels
        self.cache_dir = cache_dir
        
        #create classifier
        self.config = XLNetConfig.from_pretrained(self.language.value, num_labels=num_labels)
        self.model = XLNetForSequenceClassification(self.config)
        
    def fit(
        self,
        token_ids,
        input_mask,
        labels,
        token_type_ids=None,
        num_gpus=None,
        num_epochs=1,
        batch_size=8,
        lr=5e-5,
        adam_eps=1e-8,
        warmup_steps=0,
        weight_decay=0.0,
        max_grad_norm=1.0,
        verbose=True,
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
            num_gpus (int, optional): The number of gpus to use.
                                      If None is specified, all available GPUs
                                      will be used. Defaults to None.
            num_epochs (int, optional): Number of training epochs.
                Defaults to 1.
            batch_size (int, optional): Training batch size. Defaults to 8.
            lr (float): Learning rate of the Adam optimizer. Defaults to 5e-5.
            warmup_proportion (float, optional): Proportion of training to
                perform linear learning rate warmup for. E.g., 0.1 = 10% of
                training. Defaults to None.
            verbose (bool, optional): If True, shows the training progress and
                loss values. Defaults to True.
        """
        
        device = get_device("cpu" if num_gpus == 0 or not torch.cuda.is_available() else "gpu")
        self.model = move_to_device(self.model, device, num_gpus)
        
        # define optimizer and model parameters
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [
                    p
                    for n, p in param_optimizer
                    if not any(nd in n for nd in no_decay)
                ],
                'weight_decay': weight_decay
            },
            {
                'params': [
                    p for n, p in param_optimizer
                    if any(nd in n for nd in no_decay)
                ], 
                'weight_decay': 0.0
            }
        ]
        
        num_examples = len(token_ids)
        num_batches = int(num_examples/batch_size)
        num_train_optimization_steps = num_batches * num_epochs
        
        optimizer = AdamW(optimizer_grouped_parameters, lr=lr, eps=adam_eps)
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=warmup_steps, t_total=num_train_optimization_steps)
        
        global_step = 0
        tr_loss, logging_loss = 0.0, 0.0
        self.model.train()
        optimizer.zero_grad()
        for epoch in range(num_epochs):
            for step in range(num_batches):
              
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

                token_type_ids_batch = torch.tensor(
                        token_type_ids[start:end],
                        dtype=torch.long,
                        device=device,
                )
                
                outputs = self.model(
                    input_ids=x_batch,
                    token_type_ids=token_type_ids_batch,
                    attention_mask=mask_batch,
                    labels=y_batch,
               ) 
                
                loss = outputs[0]  # model outputs are always tuple in pytorch-transformers

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)

                tr_loss += loss.item()
                scheduler.step()  # Update learning rate schedule
                optimizer.step()
                
                optimizer.zero_grad()

                global_step += 1
                
                if verbose:
                    if step % ((num_batches // 10) + 1) == 0:
                        print(
                            "epoch:{}/{}; batch:{}->{}/{}; loss:{:.6f}".format(
                                epoch + 1,
                                num_epochs,
                                step + 1,
                                min(step + 1 + num_batches // 10, num_batches),
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
        
        device = get_device("cpu" if num_gpus == 0 or not torch.cuda.is_available() else "gpu")
        self.model = move_to_device(self.model, device, num_gpus)
        
        self.model.eval()
        preds = []
        
        with tqdm(total=len(token_ids)) as pbar:
            for i in range(0, len(token_ids), batch_size):
                start = i
                end = start + batch_size
                x_batch = torch.tensor(
                    token_ids[start:end], dtype=torch.long, device=device
                )
                mask_batch = torch.tensor(
                    input_mask[start:end], dtype=torch.long, device=device
                )

                token_type_ids_batch = torch.tensor(
                        token_type_ids[start:end],
                        dtype=torch.long,
                        device=device,
                )
                
                with torch.no_grad():
                    pred_batch = self.model(
                        input_ids=x_batch,
                        token_type_ids=token_type_ids_batch,
                        attention_mask=mask_batch,
                        labels=None
                    )
                    preds.append(pred_batch[0].cpu())
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
