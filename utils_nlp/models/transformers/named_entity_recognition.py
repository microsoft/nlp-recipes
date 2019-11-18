# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import logging
import numpy as np
import torch
import torch.nn as nn

from collections import Iterable
from torch.utils.data import TensorDataset
from transformers.modeling_bert import BERT_PRETRAINED_MODEL_ARCHIVE_MAP, BertForTokenClassification
from utils_nlp.common.pytorch_utils import get_device
from utils_nlp.models.transformers.common import MAX_SEQ_LEN, TOKENIZER_CLASS, Transformer
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler


TC_MODEL_CLASS = {k: BertForTokenClassification for k in BERT_PRETRAINED_MODEL_ARCHIVE_MAP}


class TokenClassificationProcessor:
    """
    Process raw dataset for training and testing.

    Args:
        model_name (str, optional): The pretained model name.
            Defaults to "bert-base-cased".
        to_lower (bool, optional): Lower case text input.
            Defaults to False.
        cache_dir (str, optional): The default folder for saving cache files.
            Defaults to ".".
    """

    def __init__(self, model_name="bert-base-cased", to_lower=False, cache_dir="."):
        self.model_name = model_name
        self.to_lower = to_lower
        self.cache_dir = cache_dir
        self.tokenizer = TOKENIZER_CLASS[model_name].from_pretrained(
            model_name, do_lower_case=to_lower, cache_dir=cache_dir, output_loading_info=False
        )

    @staticmethod
    def get_inputs(batch, model_name, train_mode=True):
        """
        Produce a dictionary object for model training or prediction.

        Args:
            model_name (str): The pretained model name.
            train_mode (bool, optional): Whether it's for model training. Set it to False if
                it's for testing and it won't have the 'labels' data field.
                Defaults to True, for model training.

        Returns:
            dict: A dictionary object contains all needed information for training or testing.
        """

        if model_name.split("-")[0] not in ["bert"]:
            raise ValueError("Model not supported: {}".format(model_name))

        if train_mode:
            return {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
        else:
            return {"input_ids": batch[0], "attention_mask": batch[1]}

    @staticmethod
    def create_label_map(label_lists, trailing_piece_tag="X"):
        """
        Create a dictionary object to map a label (str) to an ID (int).

        Args:
            label_lists (list): A list of label lists. Each element is a list of labels
                which presents class of each token.
            trailing_piece_tag (str, optional): Tag used to label trailing word pieces.
                Defaults to "X".

        Returns:
            dict: A dictionary object to map a label (str) to an ID (int).
        """

        label_set = set()
        for labels in label_lists:
            label_set.update(labels)

        label_map = {label: i for i, label in enumerate(label_set)}

        if trailing_piece_tag not in label_set:
            label_map[trailing_piece_tag] = len(label_set)
        return label_map

    def preprocess_for_bert(
        self, text, max_len=MAX_SEQ_LEN, labels=None, label_map=None, trailing_piece_tag="X"
    ):
        """
        Tokenize and preprocesses input word lists, involving the following steps
            0. WordPiece tokenization.
            1. Convert string tokens to token ids.
            2. Convert input labels to label ids, if labels and label_map are
                provided.
            3. If a word is tokenized into multiple pieces of tokens by the
                WordPiece tokenizer, label the extra tokens with
                trailing_piece_tag.
            4. Pad or truncate input text according to max_seq_length
            5. Create input_mask for masking out padded tokens.

        Args:
            text (list): List of lists. Each sublist is a list of words in an
                input sentence.
            max_len (int, optional): Maximum length of the list of
                tokens. Lists longer than this are truncated and shorter
                ones are padded with "O"s. Default value is BERT_MAX_LEN=512.
            labels (list, optional): List of word label lists. Each sublist
                contains labels corresponding to the input word list. The lengths
                of the label list and word list must be the same. Default
                value is None.
            label_map (dict, optional): Dictionary for mapping original token
                labels (which may be string type) to integers. Default value
                is None.
            trailing_piece_tag (str, optional): Tag used to label trailing
                word pieces. For example, "criticize" is broken into "critic"
                and "##ize", "critic" preserves its original label and "##ize"
                is labeled as trailing_piece_tag. Default value is "X".

        Returns:
            TensorDataset: A TensorDataset containing the following four tensors.
                1. input_ids_all: Tensor. Each sublist contains numerical values,
                    i.e. token ids, corresponding to the tokens in the input 
                    text data.
                2. input_mask_all: Tensor. Each sublist contains the attention
                    mask of the input token id list, 1 for input tokens and 0 for
                    padded tokens, so that padded tokens are not attended to.
                3. trailing_token_mask_all: Tensor. Each sublist is
                    a boolean list, True for the first word piece of each
                    original word, False for the trailing word pieces,
                    e.g. "##ize". This mask is useful for removing the
                    predictions on trailing word pieces, so that each
                    original word in the input text has a unique predicted
                    label.
                4. label_ids_all: Tensor, each sublist contains token labels of
                    a input sentence/paragraph, if labels is provided. If the
                    `labels` argument is not provided, it will not return this tensor.
        """

        def _is_iterable_but_not_string(obj):
            return isinstance(obj, Iterable) and not isinstance(obj, str)

        if max_len > MAX_SEQ_LEN:
            logging.warning(
                "Setting max_len to max allowed sequence length: {}".format(MAX_SEQ_LEN)
            )
            max_len = MAX_SEQ_LEN

        if not _is_iterable_but_not_string(text):
            # The input text must be an non-string Iterable
            raise ValueError("Input text must be an iterable and not a string.")
        else:
            # If the input text is a single list of words, convert it to
            # list of lists for later iteration
            if not _is_iterable_but_not_string(text[0]):
                text = [text]

        if labels is not None:
            if not _is_iterable_but_not_string(labels):
                raise ValueError("labels must be an iterable and not a string.")
            else:
                if not _is_iterable_but_not_string(labels[0]):
                    labels = [labels]

        label_available = True
        if labels is None:
            label_available = False
            # create an artificial label list for creating trailing token mask
            labels = [["O"] * len(t) for t in text]

        input_ids_all = []
        input_mask_all = []
        label_ids_all = []
        trailing_token_mask_all = []

        for t, t_labels in zip(text, labels):
            if len(t) != len(t_labels):
                raise ValueError(
                    "The number of words is {0}, but the number of labels is {1}.".format(
                        len(t), len(t_labels)
                    )
                )

            new_labels = []
            new_tokens = []
            for word, tag in zip(t, t_labels):
                sub_words = self.tokenizer.tokenize(word)
                for count, sub_word in enumerate(sub_words):
                    if count > 0:
                        tag = trailing_piece_tag
                    new_labels.append(tag)
                    new_tokens.append(sub_word)

            if len(new_tokens) > max_len:
                logging.warn(
                    "Text after tokenization with length {} has been truncated".format(
                        len(new_tokens)
                    )
                )
                new_tokens = new_tokens[:max_len]
                new_labels = new_labels[:max_len]
            input_ids = self.tokenizer.convert_tokens_to_ids(new_tokens)

            # The mask has 1 for real tokens and 0 for padding tokens.
            # Only real tokens are attended to.
            input_mask = [1.0] * len(input_ids)

            # Zero-pad up to the max sequence length.
            padding = [0.0] * (max_len - len(input_ids))
            label_padding = ["O"] * (max_len - len(input_ids))

            input_ids += padding
            input_mask += padding
            new_labels += label_padding

            trailing_token_mask_all.append(
                [True if label != trailing_piece_tag else False for label in new_labels]
            )

            if label_map:
                label_ids = [label_map[label] for label in new_labels]
            else:
                label_ids = new_labels

            input_ids_all.append(input_ids)
            input_mask_all.append(input_mask)
            label_ids_all.append(label_ids)

        if label_available:
            td = TensorDataset(
                torch.tensor(input_ids_all, dtype=torch.long),
                torch.tensor(input_mask_all, dtype=torch.long),
                torch.tensor(trailing_token_mask_all, dtype=torch.bool),
                torch.tensor(label_ids_all, dtype=torch.long),
            )
        else:
            td = TensorDataset(
                torch.tensor(input_ids_all, dtype=torch.long),
                torch.tensor(input_mask_all, dtype=torch.long),
                torch.tensor(trailing_token_mask_all, dtype=torch.bool),
            )
        return td

    def create_dataloader_from_dataset(
        self,
        dataset,
        shuffle=False,
        batch_size=32,
        num_gpus=None,
        distributed=False
    ):
        if num_gpus is None:
            num_gpus = torch.cuda.device_count()

        batch_size = batch_size * max(1, num_gpus)

        if distributed:
            sampler = DistributedSampler(dataset)
        else:
            sampler = RandomSampler(dataset) if shuffle else SequentialSampler(dataset)

        return DataLoader(dataset, sampler=sampler, batch_size=batch_size)



class TokenClassifier(Transformer):
    """
    A wrapper for token classification use case based on Transformer.

    Args:
        model_name (str, optional): The pretained model name.
            Defaults to "bert-base-cased".
        num_labels (int, optional): The number of labels.
            Defaults to 2.
        cache_dir (str, optional): The default folder for saving cache files.
            Defaults to ".".
    """

    def __init__(self, model_name="bert-base-cased", num_labels=2, cache_dir="."):
        super().__init__(
            model_class=TC_MODEL_CLASS,
            model_name=model_name,
            num_labels=num_labels,
            cache_dir=cache_dir,
        )

    @staticmethod
    def list_supported_models():
        return list(TC_MODEL_CLASS)

    def fit(
        self,
        train_dataloader,
        num_epochs=1,
        num_gpus=None,
        local_rank=-1,
        weight_decay=0.0,
        learning_rate=5e-5,
        adam_epsilon=1e-8,
        warmup_steps=0,
        verbose=True,
        seed=None,
    ):
        """
        Fit the TokenClassifier model using the given training dataset.

        Args:
            train_dataloader (DataLoader): DataLoader instance for training.
            num_epochs (int, optional): Number of training epochs.
                Defaults to 1.
            num_gpus (int, optional): The number of GPUs to use. If None, all available GPUs will
                be used. If set to 0 or GPUs are not available, CPU device will
                be used. Defaults to None.
            local_rank (int, optional): Whether need to do distributed training.
                Defaults to -1, no distributed training.
            weight_decay (float, optional): Weight decay rate.
                Defaults to 0.
            learning_rate (float, optional): The learning rate.
                Defaults to 5e-5.
            adam_espilon (float, optional): The 'eps' parameter for the 'AdamW' optimizer.
                Defaults to 1e-8.
            warmup_steps (int, optional): Number of warmup steps for 'WarmupLinearSchedule'.
                Defaults to 0.
            verbose (bool, optional): Verbose model.
                Defaults to False.
            seed (int, optional): The seed for the transformers.
                Defaults to None, use the default seed.
        """

        device, num_gpus = get_device(num_gpus=num_gpus, local_rank=local_rank)
        if isinstance(self.model, nn.DataParallel):
            self.model.module.to(device)
        else:
            self.model.to(device)

        super().fine_tune(
            train_dataloader=train_dataloader,
            get_inputs=TokenClassificationProcessor.get_inputs,
            device=device,
            n_gpu=num_gpus,
            num_train_epochs=num_epochs,
            weight_decay=weight_decay,
            learning_rate=learning_rate,
            adam_epsilon=adam_epsilon,
            warmup_steps=warmup_steps,
            verbose=verbose,
            seed=seed,
        )

    def predict(
        self,
        eval_dataloader,
        num_gpus=None,
        verbose=True
    ):
        """
        Test on an evaluation dataset and get the token label predictions.

        Args:
            eval_dataset (TensorDataset): A TensorDataset for evaluation.
            num_gpus (int, optional): The number of GPUs to use. If None, all available GPUs will
                be used. If set to 0 or GPUs are not available, CPU device will
                be used. Defaults to None.
            verbose (bool, optional): Verbose model.
                Defaults to False.

        Returns:
            ndarray: Numpy ndarray of raw predictions. The shape of the ndarray is
            [number_of_examples, sequence_length, number_of_labels]. Each
            value in the ndarray is not normalized. Post-process will be needed
            to get the probability for each class label.
        """

        device, num_gpus = get_device(num_gpus=num_gpus, local_rank=-1)
        if isinstance(self.model, nn.DataParallel):
            self.model.module.to(device)
        else:
            self.model.to(device)
        
        preds = list(
            super().predict(
                eval_dataloader=eval_dataloader,
                get_inputs=TokenClassificationProcessor.get_inputs,
                device=device,
                verbose=verbose
            )
        )
        preds_np = np.concatenate(preds)
        return preds_np

    def get_predicted_token_labels(self, predictions, label_map, dataset):
        """
        Post-process the raw prediction values and get the class label for each token.

        Args:
            predictions (ndarray): A numpy ndarray produced from the `predict` function call.
                The shape of the ndarray is [number_of_examples, sequence_length, number_of_labels].
            label_map (dict): A dictionary object to map a label (str) to an ID (int). 
                dataset (TensorDataset): The TensorDataset for evaluation.
            dataset (Dataset): The test Dataset instance.

        Returns:
            list: A list of lists. The size of the retured list is the number of testing samples.
            Each sublist represents the predicted label for each token. 
        """

        num_samples = len(dataset.tensors[0])
        if num_samples != predictions.shape[0]:
            raise ValueError(
                "Predictions have {0} samples, but got {1} samples in dataset".format(
                    predictions.shape[0], num_samples
                )
            )

        label_id2str = {v: k for k, v in label_map.items()}
        attention_mask_all = dataset.tensors[1].data.numpy()
        trailing_mask_all = dataset.tensors[2].data.numpy()
        seq_len = len(trailing_mask_all[0])
        labels = []

        for idx in range(num_samples):
            seq_probs = predictions[idx]
            attention_mask = attention_mask_all[idx]
            trailing_mask = trailing_mask_all[idx]
            one_sample = []

            for sid in range(seq_len):
                if attention_mask[sid] == 0:
                    break

                if not trailing_mask[sid]:
                    continue

                label_id = seq_probs[sid].argmax()
                one_sample.append(label_id2str[label_id])
            labels.append(one_sample)
        return labels

    def get_true_test_labels(self, label_map, dataset):
        """
        Get the true testing label values.

        Args:
            label_map (dict): A dictionary object to map a label (str) to an ID (int). 
                dataset (TensorDataset): The TensorDataset for evaluation.
            dataset (Dataset): The test Dataset instance.

        Returns:
            list: A list of lists. The size of the retured list is the number of testing samples.
            Each sublist represents the predicted label for each token. 
        """

        num_samples = len(dataset.tensors[0])
        label_id2str = {v: k for k, v in label_map.items()}
        attention_mask_all = dataset.tensors[1].data.numpy()
        trailing_mask_all = dataset.tensors[2].data.numpy()
        label_ids_all = dataset.tensors[3].data.numpy()
        seq_len = len(trailing_mask_all[0])
        labels = []

        for idx in range(num_samples):
            attention_mask = attention_mask_all[idx]
            trailing_mask = trailing_mask_all[idx]
            label_ids = label_ids_all[idx]
            one_sample = []

            for sid in range(seq_len):
                if attention_mask[sid] == 0:
                    break

                if not trailing_mask[sid]:
                    continue

                label_id = label_ids[sid]
                one_sample.append(label_id2str[label_id])
            labels.append(one_sample)
        return labels