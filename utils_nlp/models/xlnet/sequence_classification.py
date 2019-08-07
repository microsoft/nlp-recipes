import numpy as np
import torch
import torch.nn as nn
from pytorch_transformers import (WEIGHTS_NAME, XLNetConfig,XLNetForSequenceClassification,XLNetTokenizer)
from tqdm import tqdm
from torch.utils.data import (
    DataLoader,
    RandomSampler,
    SequentialSampler,
    TensorDataset,
)
from pytorch_transformers import AdamW, WarmupLinearSchedule
from utils_nlp.common.pytorch_utils import get_device, move_to_device

class XLNetSequenceClassifier:
    """XLNet-based sequence classifier"""
    
    def __init__(self, language='xlnet-base-cased', num_labels=5, cache_dir='.'):
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
        self.config = XLNetConfig.from_pretrained(
            self.language, cache_dir=cache_dir, num_labels=num_labels
        )
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
        adam_eps=1e-8
        warmup_steps=0,
        weight_decay=0.0,
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
        
        

        