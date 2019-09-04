# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# This script reuses some code from
# https://github.com/huggingface/pytorch-transformers/blob/067923d3267325f525f4e46f357360c191ba562e/examples/run_squad.py


import os
import logging
from tqdm import tqdm, trange
import numpy as np
import random

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler

import horovod.torch as hvd

# from tensorboardX import SummaryWriter

from pytorch_transformers import AdamW, WarmupLinearSchedule
from pytorch_transformers.modeling_bert import BertConfig, BertForQuestionAnswering

from utils_nlp.models.bert.common import Language
from utils_nlp.common.pytorch_utils import get_device, move_to_device

from utils_nlp.models.bert.qa_utils import QAResult

from utils_nlp.azureml.azureml_bert_util import DistributedCommunicator, warmup_linear, adjust_gradient_accumulation_steps
from azureml.core.run import Run


logger = logging.getLogger(__name__)


class BERTQAExtractor:
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

    def __init__(self, language=Language.ENGLISH, cache_dir=".", load_model_from_dir=None):

        self.language = language
        self.cache_dir = cache_dir
        self.load_model_from_dir = load_model_from_dir

        if load_model_from_dir is None:
            config = BertConfig.from_pretrained(language.value)
            self.model = BertForQuestionAnswering.from_pretrained(language.value, config=config)
        else:
            logger.info("Loading cached model from {}".format(load_model_from_dir))
            config = BertConfig.from_pretrained(load_model_from_dir)
            self.model = BertForQuestionAnswering.from_pretrained(
                load_model_from_dir, config=config
            )

    def fit(
        self,
        features,
        num_gpus=None,
        num_epochs=1,
        batch_size=32,
        learning_rate=2e-5,
        warmup_proportion=None,
        max_grad_norm=1.0,
        gradient_accumulation_steps=1,
        cache_model=False,
        overwrite_model=False,
        distributed=True
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
        ##dist
        step_per_log = 100
        is_master = False

        if distributed:
            hvd.init()
            run = Run.get_context()

            rank  = hvd.rank()
            local_rank = hvd.local_rank()
            world_size = hvd.size()

            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
            is_master = rank == 0

            self.cache_dir = self.cache_dir + "/distributed_" + str(rank)

            self.model = self.model.to(device)

        else:
            hvd.init()
            local_rank = hvd.local_rank()
            world_size = hvd.size()

            os.environ['MASTER_ADDR'] = '127.0.0.1'
            os.environ['MASTER_PORT'] = '29500'

            torch.distributed.init_process_group(
                backend="nccl",
                rank=local_rank,
                world_size=world_size
            )
            
#             world_size = torch.distributed.get_world_size()
#             local_rank = torch.distributed.get_rank()
            # device = get_device("cpu" if num_gpus == 0 or not torch.cuda.is_available() else "gpu")
            # self.model = move_to_device(self.model, device, num_gpus)
            
            device = torch.device('cuda', local_rank)
            self.model = self.model.to(device)
            
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model,
#                 device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=True
                )


        ##dist ends

        RANDOM_SEED = 42
        random.seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)
        torch.manual_seed(RANDOM_SEED)


        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)

        all_start_positions = torch.tensor([f.start_position for f in features], dtype=torch.long)
        all_end_positions = torch.tensor([f.end_position for f in features], dtype=torch.long)
        train_dataset = TensorDataset(
            all_input_ids, all_input_mask, all_segment_ids, all_start_positions, all_end_positions
        )
        ##dist
        if distributed:
            train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
        else:
            train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=local_rank)
        ##dist ends
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size)

        ##dist
        t_total = len(train_dataloader) // gradient_accumulation_steps * num_epochs
        ##dist ends

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
        optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=1e-6)


        if distributed:
            optimizer = hvd.DistributedOptimizer(
                optimizer,
                named_parameters=self.model.named_parameters(),
                backward_passes_per_step=gradient_accumulation_steps)

            hvd.broadcast_parameters(self.model.state_dict(), root_rank=0)
            hvd.broadcast_optimizer_state(optimizer, root_rank=0)

        if warmup_proportion:
            warmup_steps = t_total * warmup_proportion
        else:
            warmup_steps = 0

        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=warmup_steps, t_total=t_total)

        global_step = 0
        tr_loss = 0.0
        # self.model.zero_grad()
        self.model.train()
        train_iterator = trange(int(num_epochs), desc="Epoch")
        for _ in train_iterator:
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration", mininterval=60)):
                batch = tuple(t.to(device) for t in batch)
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2],
                    "start_positions": batch[3],
                    "end_positions": batch[4],
                }

                outputs = self.model(**inputs)
                loss = outputs[0]  # model outputs are always tuple in pytorch-transformers

                if distributed:
                    loss = loss / gradient_accumulation_steps
                else:
                    loss = loss.mean()  # mean() to average on multi-gpu parallel (not distributed) training
                    loss = loss / gradient_accumulation_steps

                loss.backward()

                if not distributed:
                    ##TODO: should this be moved to after gradient accmulation?
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)

                tr_loss += loss.item()

                global_step += 1

                if (global_step + 1) % gradient_accumulation_steps == 0:
                    if distributed:
                        optimizer.synchronize()
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                        with optimizer.skip_synchronize():
                            optimizer.step()
                    else:
                        optimizer.step()

                    scheduler.step()  # Update learning rate schedule
                    self.model.zero_grad()

                if (global_step + 1) % step_per_log == 0:
                    if distributed and is_master:
                        run.log('train_loss', np.float(tr_loss / step_per_log))
                    else:
                        logger.info(" global_step = %s, train loss = %s", global_step, tr_loss / step_per_log * gradient_accumulation_steps)
                    tr_loss = 0


        if cache_model and (not distributed or is_master):
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

        device = get_device("cpu" if num_gpus == 0 or not torch.cuda.is_available() else "gpu")
        self.model = move_to_device(self.model, device, num_gpus)

        # score
        self.model.eval()

        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)

        # This index is used to find the original data sample each
        # prediction comes from and add the unique_id to the prediction
        # results.
        # Don't use the unique_id directly because it could be string.
        all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
        test_dataset = TensorDataset(
            all_input_ids, all_input_mask, all_segment_ids, all_example_index
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

                outputs = self.model(**inputs)

            for i, example_index in enumerate(example_indices):
                test_feature = features[example_index.item()]
                unique_id = int(test_feature.unique_id)

                result = QAResult(
                    unique_id=unique_id,
                    start_logits=outputs[0][i].detach().cpu().tolist(),
                    end_logits=outputs[1][i].detach().cpu().tolist(),
                )
                all_results.append(result)
        torch.cuda.empty_cache()

        return all_results
