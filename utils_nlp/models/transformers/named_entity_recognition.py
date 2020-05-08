# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import logging
from collections import Iterable

import numpy as np
import torch
from torch.utils.data import TensorDataset
from transformers import (
    MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING,
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
)

from utils_nlp.common.pytorch_utils import compute_training_steps
from utils_nlp.models.transformers.common import MAX_SEQ_LEN, Transformer

supported_models = [
    list(x.pretrained_config_archive_map)
    for x in MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING
]
supported_models = sorted([x for y in supported_models for x in y])


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
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            do_lower_case=to_lower,
            cache_dir=cache_dir,
            output_loading_info=False,
        )

    @staticmethod
    def get_inputs(batch, device, model_name, train_mode=True):
        """
        Creates an input dictionary given a model name.

        Args:
            batch (tuple): A tuple containing input ids, attention mask,
                segment ids, and labels tensors.
            device (torch.device): A PyTorch device.
            model_name (bool): Model name used to format the inputs.
            train_mode (bool, optional): Training mode flag.
                Defaults to True.

        Returns:
            dict: Dictionary containing input ids, segment ids, masks, and labels.
                Labels are only returned when train_mode is True.
        """
        batch = tuple(t.to(device) for t in batch)
        if model_name in supported_models:
            if train_mode:
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "labels": batch[3],
                }
            else:
                inputs = {"input_ids": batch[0], "attention_mask": batch[1]}

            # distilbert doesn't support segment ids
            if model_name.split("-")[0] not in ["distilbert"]:
                inputs["token_type_ids"] = batch[2]

            return inputs
        else:
            raise ValueError("Model not supported: {}".format(model_name))

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

        unique_labels = sorted(set([x for y in label_lists for x in y]))
        label_map = {label: i for i, label in enumerate(unique_labels)}

        if trailing_piece_tag not in unique_labels:
            label_map[trailing_piece_tag] = len(unique_labels)

        return label_map

    def preprocess(
        self,
        text,
        max_len=MAX_SEQ_LEN,
        labels=None,
        label_map=None,
        trailing_piece_tag="X",
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

        logging.warn(
            "Token lists with length > {} will be truncated".format(MAX_SEQ_LEN)
        )

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
                    "Num of words and num of labels should be the same {0}!={1}".format(
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
                torch.LongTensor(input_ids_all),
                torch.LongTensor(input_mask_all),
                torch.LongTensor(trailing_token_mask_all),
                torch.LongTensor(label_ids_all),
            )
        else:
            td = TensorDataset(
                torch.LongTensor(input_ids_all),
                torch.LongTensor(input_mask_all),
                torch.LongTensor(trailing_token_mask_all),
            )
        return td


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
        config = AutoConfig.from_pretrained(
            model_name, num_labels=num_labels, cache_dir=cache_dir
        )
        model = AutoModelForTokenClassification.from_pretrained(
            model_name, cache_dir=cache_dir, config=config, output_loading_info=False
        )
        super().__init__(model_name=model_name, model=model, cache_dir=cache_dir)

    @staticmethod
    def list_supported_models():
        return supported_models

    def fit(
        self,
        train_dataloader,
        num_epochs=1,
        max_steps=-1,
        gradient_accumulation_steps=1,
        num_gpus=None,
        gpu_ids=None,
        local_rank=-1,
        weight_decay=0.0,
        learning_rate=5e-5,
        adam_epsilon=1e-8,
        warmup_steps=0,
        fp16=False,
        fp16_opt_level="O1",
        checkpoint_state_dict=None,
        verbose=True,
        seed=None,
    ):
        """
        Fine-tunes a pre-trained sequence classification model.

        Args:
            train_dataloader (Dataloader): A PyTorch DataLoader to be used for training.
            num_epochs (int, optional): Number of training epochs. Defaults to 1.
            max_steps (int, optional): Total number of training steps.
                If set to a positive value, it overrides num_epochs.
                Otherwise, it's determined by the dataset length,
                gradient_accumulation_steps, and num_epochs.
                Defualts to -1.
            gradient_accumulation_steps (int, optional): Number of steps to accumulate
                before performing a backward/update pass.
                Default to 1.
            num_gpus (int, optional): The number of GPUs to use.
                If None, all available GPUs will be used.
                If set to 0 or GPUs are not available, CPU device will be used.
                Defaults to None.
            gpu_ids (list): List of GPU IDs to be used.
                If set to None, the first num_gpus GPUs will be used.
                Defaults to None.
            local_rank (int, optional): Local_rank for distributed training on GPUs.
                Defaults to -1, which means non-distributed training.
            weight_decay (float, optional): Weight decay to apply after each
                parameter update.
                Defaults to 0.0.
            learning_rate (float, optional):  Learning rate of the AdamW optimizer.
                Defaults to 5e-5.
            adam_epsilon (float, optional): Epsilon of the AdamW optimizer.
                Defaults to 1e-8.
            warmup_steps (int, optional): Number of steps taken to increase learning
                rate from 0 to `learning rate`. Defaults to 0.
            fp16 (bool): Whether to use 16-bit mixed precision through Apex
                Defaults to False
            fp16_opt_level (str): Apex AMP optimization level for fp16.
                One of in ['O0', 'O1', 'O2', and 'O3']
                See https://nvidia.github.io/apex/amp.html"
                Defaults to "01"
            checkpoint_state_dict (dict): Checkpoint states of model and optimizer.
                If specified, the model and optimizer's parameters are loaded using
                checkpoint_state_dict["model"] and checkpoint_state_dict["optimizer"]
                Defaults to None.
            verbose (bool, optional): Whether to print out the training log.
                Defaults to True.
            seed (int, optional): Random seed used to improve reproducibility.
                Defaults to None.
        """

        # init device and optimizer
        device, num_gpus, amp = self.prepare_model_and_optimizer(
            num_gpus=num_gpus,
            gpu_ids=gpu_ids,
            local_rank=local_rank,
            weight_decay=weight_decay,
            learning_rate=learning_rate,
            adam_epsilon=adam_epsilon,
            fp16=fp16,
            fp16_opt_level=fp16_opt_level,
            checkpoint_state_dict=checkpoint_state_dict,
        )

        # compute the max number of training steps
        max_steps = compute_training_steps(
            dataloader=train_dataloader,
            num_epochs=num_epochs,
            max_steps=max_steps,
            gradient_accumulation_steps=gradient_accumulation_steps,
        )

        # init scheduler
        scheduler = Transformer.get_default_scheduler(
            optimizer=self.optimizer,
            warmup_steps=warmup_steps,
            num_training_steps=max_steps,
        )

        # fine tune
        super().fine_tune(
            train_dataloader=train_dataloader,
            get_inputs=TokenClassificationProcessor.get_inputs,
            device=device,
            num_gpus=num_gpus,
            max_steps=max_steps,
            gradient_accumulation_steps=gradient_accumulation_steps,
            optimizer=self.optimizer,
            scheduler=scheduler,
            fp16=fp16,
            amp=amp,
            local_rank=local_rank,
            verbose=verbose,
            seed=seed,
        )

    def predict(self, test_dataloader, num_gpus=None, gpu_ids=None, verbose=True):
        """
        Scores a dataset using a fine-tuned model and a given dataloader.

        Args:
            test_dataloader (DataLoader): DataLoader for scoring the data.
            num_gpus (int, optional): The number of GPUs to use.
                If None, all available GPUs will be used. If set to 0 or GPUs are
                not available, CPU device will be used.
                Defaults to None.
            gpu_ids (list): List of GPU IDs to be used.
                If set to None, the first num_gpus GPUs will be used.
                Defaults to None.
            verbose (bool, optional): Whether to print out the training log.
                Defaults to True.

        Returns
            1darray: numpy array of predicted label indices.
        """

        preds = list(
            super().predict(
                eval_dataloader=test_dataloader,
                get_inputs=TokenClassificationProcessor.get_inputs,
                num_gpus=num_gpus,
                gpu_ids=gpu_ids,
                verbose=verbose,
            )
        )
        preds = np.concatenate(preds)
        return preds

    def get_predicted_token_labels(self, predictions, label_map, dataset):
        """
        Post-process the raw prediction values and get the class label for each token.

        Args:
            predictions (ndarray): A numpy ndarray produced from the `predict`
                function call. The shape of the ndarray is:
                [number_of_examples, sequence_length, number_of_labels].
            label_map (dict): A dictionary object to map a label (str) to an ID (int).
                dataset (TensorDataset): The TensorDataset for evaluation.
            dataset (Dataset): The test Dataset instance.

        Returns:
            list: A list of lists. The size of the retured list is the number of
                testing samples.
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

                if not bool(trailing_mask[sid]):
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
            list: A list of lists. The size of the retured list is the number
                of testing samples.
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
