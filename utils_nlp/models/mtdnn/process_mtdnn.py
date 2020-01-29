# coding=utf-8
# Copyright (c) Microsoft. All rights reserved.
import logging
import os

from torch.utils.data import BatchSampler, DataLoader, Dataset

from utils_nlp.models.mtdnn.common.types import TaskType
from utils_nlp.models.mtdnn.configuration_mtdnn import MTDNNConfig
from utils_nlp.models.mtdnn.dataset_mtdnn import (
    MTDNNCollater,
    MTDNNMultiTaskBatchSampler,
    MTDNNMultiTaskDataset,
    MTDNNSingleTaskDataset,
)
from utils_nlp.models.mtdnn.tasks.config import TaskDefs

logger = logging.getLogger(__name__)


class MTDNNDataPreprocess:
    def __init__(
        self,
        config: MTDNNConfig,
        task_defs: TaskDefs,
        batch_size: int,
        data_dir: str = "data/canonical_data/bert_uncased_lower",
        train_datasets_list: str = ["mnli"],
        test_datasets_list: str = ["mnli_mismatched,mnli_matched"],
        glue_format: bool = False,
        data_sort: bool = False,
    ):
        assert len(train_datasets_list) >= 1, "Train dataset list cannot be empty"
        assert len(test_datasets_list) >= 1, "Test dataset list cannot be empty"

        # Initialize class members
        self.config = config
        self.task_defs = task_defs
        self.train_datasets = train_datasets_list
        self.test_datasets = test_datasets_list
        self.data_dir = data_dir
        self.glue_format = glue_format
        self.data_sort = data_sort
        self.batch_size = batch_size
        self.tasks = {}
        self.tasks_class = {}
        self.nclass_list = []
        self.decoder_opts = []
        self.task_types = []
        self.dropout_list = []
        self.loss_types = []
        self.kd_loss_types = []
        self.train_data = self.process_train_datasets()
        self.dev_datasets_list, self.test_datasets_list = self.process_dev_test_datasets()
        self.num_all_batches = (
            self.config.epochs * len(self.train_data) // self.config.grad_accumulation_step
        )

    def process_train_datasets(self):
        """Preprocess the training sets and generate decoding and task specific training options needed to update config object
        
        Returns:
            [DataLoader] -- Multiple tasks train data ready for training
        """
        logger.info("Starting to process the training data sets")

        train_datasets = []
        for dataset in self.train_datasets:
            prefix = dataset.split("_")[0]
            if prefix in self.tasks:
                continue
            assert prefix in self.task_defs.n_class_map
            assert prefix in self.task_defs.data_type_map
            data_type = self.task_defs.data_type_map[prefix]
            nclass = self.task_defs.n_class_map[prefix]
            task_id = len(self.tasks)
            if self.config.mtl_opt > 0:
                task_id = (
                    self.tasks_class[nclass]
                    if nclass in self.tasks_class
                    else len(self.tasks_class)
                )

            task_type = self.task_defs.task_type_map[prefix]

            dopt = self.generate_decoder_opt(
                self.task_defs.enable_san_map[prefix], self.config.answer_opt
            )
            if task_id < len(self.decoder_opts):
                self.decoder_opts[task_id] = min(self.decoder_opts[task_id], dopt)
            else:
                self.decoder_opts.append(dopt)
            self.task_types.append(task_type)
            self.loss_types.append(self.task_defs.loss_map[prefix])
            self.kd_loss_types.append(self.task_defs.kd_loss_map[prefix])

            if prefix not in self.tasks:
                self.tasks[prefix] = len(self.tasks)
                if self.config.mtl_opt < 1:
                    self.nclass_list.append(nclass)

            if nclass not in self.tasks_class:
                self.tasks_class[nclass] = len(self.tasks_class)
                if self.config.mtl_opt > 0:
                    self.nclass_list.append(nclass)

            dropout_p = self.task_defs.dropout_p_map.get(prefix, self.config.dropout_p)
            self.dropout_list.append(dropout_p)

            train_path = os.path.join(self.data_dir, f"{dataset}_train.json")
            logger.info(f"Loading {train_path} as task {task_id}")
            train_data_set = MTDNNSingleTaskDataset(
                train_path,
                True,
                maxlen=self.config.max_seq_len,
                task_id=task_id,
                task_type=task_type,
                data_type=data_type,
            )
            train_datasets.append(train_data_set)
        train_collater = MTDNNCollater(
            dropout_w=self.config.dropout_w, encoder_type=self.config.encoder_type
        )
        multi_task_train_dataset = MTDNNMultiTaskDataset(train_datasets)
        multi_task_batch_sampler = MTDNNMultiTaskBatchSampler(
            train_datasets, self.config.batch_size, self.config.mix_opt, self.config.ratio
        )
        multi_task_train_data = DataLoader(
            multi_task_train_dataset,
            batch_sampler=multi_task_batch_sampler,
            collate_fn=train_collater.collate_fn,
            pin_memory=self.config.cuda,
        )

        # Update class configuration with decoder opts
        self.config = self.update_config(self.config)

        return multi_task_train_data

    def process_dev_test_datasets(self):
        """Preprocess the test sets 
        
        Returns:
            [List] -- Multiple tasks test data ready for inference
        """
        logger.info("Starting to process the testing data sets")
        dev_data_list = []
        test_data_list = []
        test_collater = MTDNNCollater(is_train=False, encoder_type=self.config.encoder_type)
        for dataset in self.test_datasets:
            prefix = dataset.split("_")[0]
            task_id = (
                self.tasks_class[self.task_defs.n_class_map[prefix]]
                if self.config.mtl_opt > 0
                else self.tasks[prefix]
            )
            task_type = self.task_defs.task_type_map[prefix]

            pw_task = False
            if task_type == TaskType.Ranking:
                pw_task = True

            assert prefix in self.task_defs.data_type_map
            data_type = self.task_defs.data_type_map[prefix]

            dev_path = os.path.join(self.data_dir, f"{dataset}_dev.json")
            dev_data = None
            if os.path.exists(dev_path):
                dev_data_set = MTDNNSingleTaskDataset(
                    dev_path,
                    False,
                    maxlen=self.config.max_seq_len,
                    task_id=task_id,
                    task_type=task_type,
                    data_type=data_type,
                )
                dev_data = DataLoader(
                    dev_data_set,
                    batch_size=self.config.batch_size_eval,
                    collate_fn=test_collater.collate_fn,
                    pin_memory=self.config.cuda,
                )
            dev_data_list.append(dev_data)

            test_path = os.path.join(self.data_dir, f"{dataset}_test.json")
            test_data = None
            if os.path.exists(test_path):
                test_data_set = MTDNNSingleTaskDataset(
                    test_path,
                    False,
                    maxlen=self.config.max_seq_len,
                    task_id=task_id,
                    task_type=task_type,
                    data_type=data_type,
                )
                test_data = DataLoader(
                    test_data_set,
                    batch_size=self.config.batch_size_eval,
                    collate_fn=test_collater.collate_fn,
                    pin_memory=self.config.cuda,
                )
            test_data_list.append(test_data)

        # Return tuple of dev and test data
        return dev_data_list, test_data_list

    def generate_decoder_opt(self, enable_san, max_opt):
        return max_opt if enable_san and max_opt < 3 else 0

    def update_config(self, config: MTDNNConfig):
        # Update configurations with options obtained from preprocessing training data
        setattr(config, "decoder_opts", self.decoder_opts)
        setattr(config, "task_types", self.task_types)
        setattr(config, "tasks_dropout_p", self.dropout_list)
        setattr(config, "loss_types", self.loss_types)
        setattr(config, "kd_loss_types", self.kd_loss_types)
        setattr(config, "tasks_nclass_list", self.nclass_list)
        return config
