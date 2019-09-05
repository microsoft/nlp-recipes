# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# This script reuses some code from
# https://github.com/huggingface/pytorch-transformers/blob/067923d3267325f525f4e46f357360c191ba562e/examples/run_squad.py


import os
import logging
from tqdm import tqdm, trange

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset

# from tensorboardX import SummaryWriter

from pytorch_transformers import AdamW, WarmupLinearSchedule
from pytorch_transformers import (
    BertConfig,
    BertForQuestionAnswering,
    XLNetConfig,
    XLNetForQuestionAnswering,
)

from utils_nlp.common.pytorch_utils import get_device, move_to_device

from utils_nlp.models.transformers.qa_utils import QAResult, QAResultExtended
from utils_nlp.models.transformers.common import ModelType


logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    "bert": (BertConfig, BertForQuestionAnswering),
    "xlnet": (XLNetConfig, XLNetForQuestionAnswering),
}


class TransformerAnswerExtractor:
    """
    Question answer extractor based on
    :class:`pytorch_transformers.modeling_bert.BertForQuestionAnswering`

    Args:
        language (Language, optional): The pre-trained model's language.
            The value of this argument determines which BERT model is
            used. See :class:`~utils_nlp.models.bert.common.Language`
            for details. Defaults to Language.ENGLISH.
        cache_dir (str, optional):  Location of BERT's cache directory.
            When calling the `fit` method, if `cache_model` is `True`,
            the fine-tuned model is saved to this directory. If `cache_dir`
            and `load_model_from_dir` are the same and `overwrite_model` is
            `False`, the fitted model is saved to "cache_dir/fine_tuned".
            Defaults to ".".
        load_model_from_dir (str, optional): Directory to load the model from.
            The directory must contain a model file "pytorch_model.bin" and a
            configuration file "config.json". Defaults to None.

    """

    def __init__(self, model_type, sub_model_type, cache_dir=".", load_model_from_dir=None):

        self.model_type = model_type
        self.sub_model_type = sub_model_type
        self.cache_dir = cache_dir
        self.load_model_from_dir = load_model_from_dir

        config_class, model_class = MODEL_CLASSES[self.model_type.value]

        if load_model_from_dir is None:
            config = config_class.from_pretrained(self.sub_model_type.value)
            self.model = model_class.from_pretrained(self.sub_model_type.value, config=config)
        else:
            logger.info("Loading cached model from {}".format(load_model_from_dir))
            config = config_class.from_pretrained(load_model_from_dir)
            self.model = model_class.from_pretrained(load_model_from_dir, config=config)

    def fit(
        self,
        features,
        num_gpus=None,
        num_epochs=1,
        batch_size=32,
        learning_rate=2e-5,
        warmup_proportion=None,
        max_grad_norm=1.0,
        cache_model=False,
        overwrite_model=False,
    ):
        """
        Fine-tune pre-trained BertForQuestionAnswering model.

        Args:
            features (list): List of QAFeatures containing features of
                training data. Use
                :meth:`utils_nlp.models.bert.common.Tokenizer.tokenize_qa`
                to generate training features. See
                :class:`~utils_nlp.models.bert.qa_utils.QAFeatures` for
                details of QAFeatures.
            num_gpus (int, optional): The number of GPUs to use.
                If None, all available GPUs will be used. Defaults to None.
            num_epochs (int, optional): Number of training epochs. Defaults
                to 1.
            batch_size (int, optional): Training batch size. Defaults to 32.
            learning_rate (float, optional):  Learning rate of the AdamW
                optimizer. Defaults to 2e-5.
            warmup_proportion (float, optional): Proportion of training to
                perform linear learning rate warmup for. E.g., 0.1 = 10% of
                training. Defaults to None.
            max_grad_norm (float, optional): Maximum gradient norm for gradient
                clipping. Defaults to 1.0.
            cache_model (bool, optional): Whether to save the fine-tuned
                model to the `cache_dir` of the answer extractor.
                If `cache_dir` and `load_model_from_dir` are the same and
                `overwrite_model` is `False`, the fitted model is saved
                to "cache_dir/fine_tuned". Defaults to False.
            overwrite_model (bool, optional): Whether to overwrite an existing model.
                If `cache_dir` and `load_model_from_dir` are the same and
                `overwrite_model` is `False`, the fitted model is saved to
                "cache_dir/fine_tuned". Defaults to False.

        """
        # tb_writer = SummaryWriter()
        device = get_device("cpu" if num_gpus == 0 or not torch.cuda.is_available() else "gpu")
        self.model = move_to_device(self.model, device, num_gpus)

        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        all_cls_index = torch.tensor([f.cls_index for f in features], dtype=torch.long)
        all_p_mask = torch.tensor([f.p_mask for f in features], dtype=torch.float)

        all_start_positions = torch.tensor([f.start_position for f in features], dtype=torch.long)
        all_end_positions = torch.tensor([f.end_position for f in features], dtype=torch.long)
        train_dataset = TensorDataset(
            all_input_ids, all_input_mask, all_segment_ids, all_start_positions, all_end_positions, all_cls_index, all_p_mask
        )

        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size)

        t_total = len(train_dataloader) * num_epochs

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.01,
            },
            {
                "params": [
                    p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=1e-8)

        if warmup_proportion:
            warmup_steps = t_total * warmup_proportion
        else:
            warmup_steps = 0

        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=warmup_steps, t_total=t_total)

        global_step = 0
        tr_loss = 0.0
        self.model.zero_grad()
        train_iterator = trange(int(num_epochs), desc="Epoch")
        for _ in train_iterator:
            for batch in tqdm(train_dataloader, desc="Iteration", mininterval=60):
                self.model.train()
                batch = tuple(t.to(device) for t in batch)
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2],
                    "start_positions": batch[3],
                    "end_positions": batch[4],
                }

                if self.model_type in [ModelType.XLNet]:
                    inputs.update({"cls_index": batch[5], "p_mask": batch[6]})

                outputs = self.model(**inputs)
                loss = outputs[0]  # model outputs are always tuple in pytorch-transformers

                loss = (
                    loss.mean()
                )  # mean() to average on multi-gpu parallel (not distributed) training

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)

                tr_loss += loss.item()
                
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                self.model.zero_grad()
                global_step += 1

        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss / global_step)

        if cache_model:
            if self.cache_dir == self.load_model_from_dir and not overwrite_model:
                output_model_dir = os.path.join(self.cache_dir, "fine_tuned")
            else:
                output_model_dir = self.cache_dir

            if not os.path.exists(self.cache_dir):
                os.makedirs(self.cache_dir)
            if not os.path.exists(output_model_dir):
                os.makedirs(output_model_dir)

            logger.info("Saving model checkpoint to %s", output_model_dir)
            # Save a trained model, configuration and tokenizer using `save_pretrained()`.
            # They can then be reloaded using `from_pretrained()`
            model_to_save = (
                self.model.module if hasattr(self.model, "module") else self.model
            )  # Take care of distributed/parallel training
            model_to_save.save_pretrained(output_model_dir)
        torch.cuda.empty_cache()

    def predict(self, features, num_gpus=None, batch_size=32):

        """
        Predicts answer start and end logits using fine-tuned
        BertForQuestionAnswering model.

        Args:
            features (list): List of QAFeatures containing features of
                testing data. Use
                :meth:`utils_nlp.models.bert.common.Tokenizer.tokenize_qa`
                to generate training features. See
                :class:`~utils_nlp.models.bert.qa_utils.QAFeatures` for
                details of QAFeatures.
            num_gpus (int, optional): The number of GPUs to use.
                If None, all available GPUs will be used. Defaults to None.
            batch_size (int, optional): Training batch size. Defaults to 32.

        Returns:
            list: List of QAResults. Each QAResult contains the unique id,
                answer start logits, and answer end logits of the tokens in
                QAFeatures.tokens of the input features. Use
                :func:`utils_nlp.models.bert.qa_utils.postprocess_answer` to
                generate the final predicted answers.
        """

        def _to_list(tensor):
            return tensor.detach().cpu().tolist()

        device = get_device("cpu" if num_gpus == 0 or not torch.cuda.is_available() else "gpu")
        self.model = move_to_device(self.model, device, num_gpus)

        # score
        self.model.eval()

        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        all_cls_index = torch.tensor([f.cls_index for f in features], dtype=torch.long)
        all_p_mask = torch.tensor([f.p_mask for f in features], dtype=torch.float)

        # This index is used to find the original data sample each
        # prediction comes from and add the unique_id to the prediction
        # results.
        # Don't use the unique_id directly because it could be string.
        all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
        test_dataset = TensorDataset(
            all_input_ids, all_input_mask, all_segment_ids, all_example_index, all_cls_index, all_p_mask
        )

        test_sampler = SequentialSampler(test_dataset)
        test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=batch_size)

        all_results = []
        for batch in tqdm(test_dataloader, desc="Evaluating"):
            batch = tuple(t.to(device) for t in batch)
            with torch.no_grad():
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2],
                }
                example_indices = batch[3]

                if self.model_type in [ModelType.XLNet]:
                    inputs.update({'cls_index': batch[4],
                                'p_mask':    batch[5]})

                outputs = self.model(**inputs)

            for i, example_index in enumerate(example_indices):
                test_feature = features[example_index.item()]
                unique_id = int(test_feature.unique_id)

                if self.model_type in [ModelType.XLNet]:
                    result = QAResultExtended(
                        unique_id=unique_id,
                        start_top_log_probs=_to_list(outputs[0][i]),
                        start_top_index=_to_list(outputs[1][i]),
                        end_top_log_probs=_to_list(outputs[2][i]),
                        end_top_index=_to_list(outputs[3][i]),
                        cls_logits=_to_list(outputs[4][i]),
                    )
                else:
                    result = QAResult(
                        unique_id=unique_id,
                        start_logits=_to_list(outputs[0][i]),
                        end_logits=_to_list(outputs[1][i]),
                    )
                all_results.append(result)
        torch.cuda.empty_cache()

        return all_results
