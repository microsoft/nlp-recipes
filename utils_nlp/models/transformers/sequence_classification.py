# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
from transformers import (
    MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from utils_nlp.common.pytorch_utils import compute_training_steps
from utils_nlp.models.transformers.common import MAX_SEQ_LEN, Transformer
from utils_nlp.models.transformers.datasets import SCDataSet, SPCDataSet

supported_models = [
    list(x.pretrained_config_archive_map)
    for x in MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING
]
supported_models = sorted([x for y in supported_models for x in y])


class Processor:
    """
    Class for preprocessing sequence classification data.

    Args:
        model_name (str, optional): Name of the model.
            Call SequenceClassifier.list_supported_models() to get all supported models.
            Defaults to "bert-base-cased".
        to_lower (bool, optional): Whether to convert all letters to lower case during
            tokenization. This is determined by if a cased model is used.
            Defaults to False, which corresponds to a cased model.
        cache_dir (str, optional): Directory to cache the tokenizer. Defaults to ".".
        output_loading_info (bool, optional): Display tokenizer loading info if True.
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

            # distilbert, bart don't support segment ids
            if model_name.split("-")[0] not in ["distilbert", "bart"]:
                inputs["token_type_ids"] = batch[2]

            return inputs
        else:
            raise ValueError("Model not supported: {}".format(model_name))

    @staticmethod
    def text_transform(text, tokenizer, max_len=MAX_SEQ_LEN):
        """
        Text transformation function for sequence classification.
        The function can be passed to a map-style PyTorch DataSet.

        Args:
            text (str): Input text.
            tokenizer (PreTrainedTokenizer): A pretrained tokenizer.
            max_len (int, optional): Max sequence length. Defaults to 512.

        Returns:
            tuple: Tuple containing input ids, attention masks, and segment ids.
        """
        if max_len > MAX_SEQ_LEN:
            print("setting max_len to max allowed seq length: {}".format(MAX_SEQ_LEN))
            max_len = MAX_SEQ_LEN
        # truncate and add CLS & SEP markers
        tokens = tokenizer.tokenize(text)[0 : max_len - 2]
        tokens = [tokenizer.cls_token] + tokens + [tokenizer.sep_token]

        # get input ids
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        # pad sequence
        input_ids = input_ids + [0] * (max_len - len(input_ids))
        # create input mask
        attention_mask = [min(1, x) for x in input_ids]
        # create segment ids
        token_type_ids = [0] * len(input_ids)

        return input_ids, attention_mask, token_type_ids

    @staticmethod
    def text_pair_transform(text_1, text_2, tokenizer, max_len=MAX_SEQ_LEN):
        """
        Text transformation function for sentence pair classification.
        The function can be passed to a map-style PyTorch DataSet.

        Args:
            text_1 (str): Input text 1.
            text_2 (str): Input text 2.
            tokenizer (PreTrainedTokenizer): A pretrained tokenizer.
            max_len (int, optional): Max sequence length. Defaults to 512.

        Returns:
            tuple: Tuple containing input ids, attention masks, and segment ids.
        """

        def _truncate_seq_pair(tokens_a, tokens_b, max_length):
            """Truncates a sequence pair in place to the maximum length."""
            # This is a simple heuristic which will always truncate the longer
            # sequence one token at a time. This makes more sense than
            # truncating an equal percent of tokens from each, since if one
            # sequence is very short then each token that's truncated likely
            # contains more information than a longer sequence.

            if not tokens_b:
                max_length += 1

            while True:
                total_length = len(tokens_a) + len(tokens_b)
                if total_length <= max_length:
                    break
                if len(tokens_a) > len(tokens_b):
                    tokens_a.pop()
                else:
                    tokens_b.pop()

            tokens_a.append(tokenizer.sep_token)

            if tokens_b:
                tokens_b.append(tokenizer.sep_token)

            return tokens_a, tokens_b

        if max_len > MAX_SEQ_LEN:
            print("setting max_len to max allowed tokens: {}".format(MAX_SEQ_LEN))
            max_len = MAX_SEQ_LEN

        tokens_1 = tokenizer.tokenize(text_1)

        tokens_2 = tokenizer.tokenize(text_2)

        tokens_1, tokens_2 = _truncate_seq_pair(tokens_1, tokens_2, max_len - 3)

        # construct token_type_ids, prefix with [0] for [CLS]
        # [0, 0, 0, 0, ... 0, 1, 1, 1, ... 1]
        token_type_ids = [0] + [0] * len(tokens_1) + [1] * len(tokens_2)
        # pad sequence
        token_type_ids = token_type_ids + [0] * (max_len - len(token_type_ids))
        # merge sentences
        tokens = [tokenizer.cls_token] + tokens_1 + tokens_2
        # convert tokens to indices
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        # pad sequence
        input_ids = input_ids + [0] * (max_len - len(input_ids))
        # create input mask
        attention_mask = [min(1, x) for x in input_ids]

        return input_ids, attention_mask, token_type_ids

    def dataset_from_dataframe(
        self, df, text_col, label_col=None, text2_col=None, max_len=MAX_SEQ_LEN
    ):
        if text2_col is None:
            return SCDataSet(
                df,
                text_col,
                label_col,
                transform=Processor.text_transform,
                tokenizer=self.tokenizer,
                max_len=max_len,
            )
        else:
            return SPCDataSet(
                df,
                text_col,
                text2_col,
                label_col,
                transform=Processor.text_pair_transform,
                tokenizer=self.tokenizer,
                max_len=max_len,
            )


class SequenceClassifier(Transformer):
    def __init__(self, model_name="bert-base-cased", num_labels=2, cache_dir="."):
        config = AutoConfig.from_pretrained(
            model_name, num_labels=num_labels, cache_dir=cache_dir
        )
        model = AutoModelForSequenceClassification.from_pretrained(
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
            get_inputs=Processor.get_inputs,
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
                get_inputs=Processor.get_inputs,
                num_gpus=num_gpus,
                gpu_ids=gpu_ids,
                verbose=verbose,
            )
        )
        preds = np.concatenate(preds)
        return np.argmax(preds, axis=1)
