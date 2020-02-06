# coding=utf-8
# Copyright (c) Microsoft. All rights reserved.

# This script reuses some code from https://github.com/huggingface/transformers


""" Model configuration """

from __future__ import absolute_import, division, print_function, unicode_literals

import copy
import json
import logging
import os
from typing import Union

import yaml

from utils_nlp.models.mtdnn.common.loss import LossCriterion
from utils_nlp.models.mtdnn.common.metrics import Metric
from utils_nlp.models.mtdnn.common.types import DataFormat, EncoderModelType, TaskType
from utils_nlp.models.mtdnn.common.vocab import Vocabulary
from utils_nlp.models.mtdnn.common.utils import MTDNNCommonUtils


logger = MTDNNCommonUtils.setup_logging()


class TaskConfig(object):
    """Base Class for Task Configurations

    Handles parameters that are common to all task configurations

    Arguments:
        object {[type]} -- [description]
    """

    def __init__(self, **kwargs: dict):
        """ Define a generic task configuration """
        logger.info("Mapping Task attributes")

        # Mapping attributes
        for key, value in kwargs.items():
            try:
                setattr(self, key, value)
            except AttributeError as err:
                logger.error(f"[ERROR] - Unable to set {key} with value {value} for {self}")
                raise err

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        return copy.deepcopy(self.__dict__)


class COLATaskConfig(TaskConfig):
    def __init__(self, kwargs: dict = {}):
        if not kwargs:
            kwargs = {
                "task_name": "cola",
                "data_format": "PremiseOnly",
                "encoder_type": "BERT",
                "dropout_p": 0.05,
                "enable_san": False,
                "metric_meta": ["ACC", "MCC"],
                "loss": "CeCriterion",
                "kd_loss": "MseCriterion",
                "n_class": 2,
                "task_type": "Classification",
            }
        super(COLATaskConfig, self).__init__(**kwargs)
        self.dropout_p = kwargs.pop("dropout_p", 0.05)


class MNLITaskConfig(TaskConfig):
    def __init__(self, kwargs: dict = {}):
        if not kwargs:
            kwargs = {
                "task_name": "mnli",
                "data_format": "PremiseAndOneHypothesis",
                "encoder_type": "BERT",
                "dropout_p": 0.3,
                "enable_san": True,
                "labels": ["contradiction", "neutral", "entailment"],
                "metric_meta": ["ACC"],
                "loss": "CeCriterion",
                "kd_loss": "MseCriterion",
                "n_class": 3,
                "split_names": [
                    "train",
                    "matched_dev",
                    "mismatched_dev",
                    "matched_test",
                    "mismatched_test",
                ],
                "task_type": "Classification",
            }
        super(MNLITaskConfig, self).__init__(**kwargs)
        self.dropout_p = kwargs.pop("dropout_p", 0.3)
        self.split_names = kwargs.pop(
            "split_names",
            ["train", "matched_dev", "mismatched_dev", "matched_test", "mismatched_test"],
        )


class MRPCTaskConfig(TaskConfig):
    def __init__(self, kwargs: dict = {}):
        if not kwargs:
            kwargs = {
                "task_name": "mrpc",
                "data_format": "PremiseAndOneHypothesis",
                "encoder_type": "BERT",
                "enable_san": True,
                "metric_meta": ["ACC", "F1"],
                "loss": "CeCriterion",
                "kd_loss": "MseCriterion",
                "n_class": 2,
                "task_type": "Classification",
            }
        super(MRPCTaskConfig, self).__init__(**kwargs)


class QNLITaskConfig(TaskConfig):
    def __init__(self, kwargs: dict = {}):
        if not kwargs:
            kwargs = {
                "task_name": "qnli",
                "data_format": "PremiseAndOneHypothesis",
                "encoder_type": "BERT",
                "enable_san": True,
                "labels": ["not_entailment", "entailment"],
                "metric_meta": ["ACC"],
                "loss": "CeCriterion",
                "kd_loss": "MseCriterion",
                "n_class": 2,
                "task_type": "Classification",
            }
        super(QNLITaskConfig, self).__init__(**kwargs)
        self.labels = kwargs.pop("labels", ["not_entailment", "entailment"])


class QQPTaskConfig(TaskConfig):
    def __init__(self, kwargs: dict = {}):
        if not kwargs:
            kwargs = {
                "task_name": "qqp",
                "data_format": "PremiseAndOneHypothesis",
                "encoder_type": "BERT",
                "enable_san": True,
                "metric_meta": ["ACC", "F1"],
                "loss": "CeCriterion",
                "kd_loss": "MseCriterion",
                "n_class": 2,
                "task_type": "Classification",
            }
        super(QQPTaskConfig, self).__init__(**kwargs)


class RTETaskConfig(TaskConfig):
    def __init__(self, kwargs: dict = {}):
        if not kwargs:
            kwargs = {
                "task_name": "rte",
                "data_format": "PremiseAndOneHypothesis",
                "encoder_type": "BERT",
                "enable_san": True,
                "labels": ["not_entailment", "entailment"],
                "metric_meta": ["ACC"],
                "loss": "CeCriterion",
                "kd_loss": "MseCriterion",
                "n_class": 2,
                "task_type": "Classification",
            }
        super(RTETaskConfig, self).__init__(**kwargs)
        self.labels = kwargs.pop("labels", ["not_entailment", "entailment"])


class SCITAILTaskConfig(TaskConfig):
    def __init__(self, kwargs: dict = {}):
        if not kwargs:
            kwargs = {
                "task_name": "scitail",
                "encoder_type": "BERT",
                "data_format": "PremiseAndOneHypothesis",
                "enable_san": True,
                "labels": ["neutral", "entails"],
                "metric_meta": ["ACC"],
                "loss": "CeCriterion",
                "kd_loss": "MseCriterion",
                "n_class": 2,
                "task_type": "Classification",
            }
        super(SCITAILTaskConfig, self).__init__(**kwargs)
        self.labels = kwargs.pop("labels", ["neutral", "entails"])


class SNLITaskConfig(TaskConfig):
    def __init__(self, kwargs: dict = {}):
        if not kwargs:
            kwargs = {
                "task_name": "snli",
                "data_format": "PremiseAndOneHypothesis",
                "encoder_type": "BERT",
                "enable_san": True,
                "labels": ["contradiction", "neutral", "entailment"],
                "metric_meta": ["ACC"],
                "loss": "CeCriterion",
                "kd_loss": "MseCriterion",
                "n_class": 3,
                "task_type": "Classification",
            }
        super(SNLITaskConfig, self).__init__(**kwargs)
        self.labels = kwargs.pop("labels", ["contradiction", "neutral", "entailment"])


class SSTTaskConfig(TaskConfig):
    def __init__(self, kwargs: dict = {}):
        if not kwargs:
            kwargs = {
                "task_name": "sst",
                "data_format": "PremiseOnly",
                "encoder_type": "BERT",
                "enable_san": False,
                "metric_meta": ["ACC"],
                "loss": "CeCriterion",
                "kd_loss": "MseCriterion",
                "n_class": 2,
                "task_type": "Classification",
            }
        super(SSTTaskConfig, self).__init__(**kwargs)


class STSBTaskConfig(TaskConfig):
    def __init__(self, kwargs: dict = {}):
        if not kwargs:
            kwargs = {
                "task_name": "stsb",
                "data_format": "PremiseAndOneHypothesis",
                "encoder_type": "BERT",
                "enable_san": false,
                "metric_meta": ["Pearson", "Spearman"],
                "n_class": 1,
                "loss": "MseCriterion",
                "kd_loss": "MseCriterion",
                "task_type": "Regression",
            }
        super(STSBTaskConfig, self).__init__(**kwargs)


class WNLITaskConfig(TaskConfig):
    def __init__(self, kwargs: dict = {}):
        if not kwargs:
            kwargs = {
                "task_name": "wnli",
                "data_format": "PremiseAndOneHypothesis",
                "encoder_type": "BERT",
                "enable_san": True,
                "metric_meta": ["ACC"],
                "loss": "CeCriterion",
                "kd_loss": "MseCriterion",
                "n_class": 2,
                "task_type": "Classification",
            }
        super(WNLITaskConfig, self).__init__(**kwargs)


class NERTaskConfig(TaskConfig):
    def __init__(self, kwargs: dict = {}):
        if not kwargs:
            kwargs = {
                "task_name": "ner",
                "data_format": "Seqence",
                "encoder_type": "BERT",
                "dropout_p": 0.3,
                "enable_san": False,
                "labels": [
                    "O",
                    "B-MISC",
                    "I-MISC",
                    "B-PER",
                    "I-PER",
                    "B-ORG",
                    "I-ORG",
                    "B-LOC",
                    "I-LOC",
                    "X",
                    "CLS",
                    "SEP",
                ],
                "metric_meta": ["SeqEval"],
                "n_class": 12,
                "loss": "SeqCeCriterion",
                "split_names": ["train", "dev", "test"],
                "task_type": "SequenceLabeling",
            }
        super(NERTaskConfig, self).__init__(**kwargs)
        self.labels = kwargs.pop(
            "labels",
            [
                "O",
                "B-MISC",
                "I-MISC",
                "B-PER",
                "I-PER",
                "B-ORG",
                "I-ORG",
                "B-LOC",
                "I-LOC",
                "X",
                "CLS",
                "SEP",
            ],
        )
        self.split_names = kwargs.pop("split_names", ["train", "dev", "test"])


class POSTaskConfig(TaskConfig):
    def __init__(self, kwargs: dict = {}):
        if not kwargs:
            kwargs = {
                "task_name": "pos",
                "data_format": "Seqence",
                "encoder_type": "BERT",
                "dropout_p": 0.1,
                "enable_san": False,
                "labels": [
                    ",",
                    "\\",
                    ":",
                    ".",
                    "''",
                    '"',
                    "(",
                    ")",
                    "$",
                    "CC",
                    "CD",
                    "DT",
                    "EX",
                    "FW",
                    "IN",
                    "JJ",
                    "JJR",
                    "JJS",
                    "LS",
                    "MD",
                    "NN",
                    "NNP",
                    "NNPS",
                    "NNS",
                    "NN|SYM",
                    "PDT",
                    "POS",
                    "PRP",
                    "PRP$",
                    "RB",
                    "RBR",
                    "RBS",
                    "RP",
                    "SYM",
                    "TO",
                    "UH",
                    "VB",
                    "VBD",
                    "VBG",
                    "VBN",
                    "VBP",
                    "VBZ",
                    "WDT",
                    "WP",
                    "WP$",
                    "WRB",
                    "X",
                    "CLS",
                    "SEP",
                ],
                "metric_meta": ["SeqEval"],
                "n_class": 49,
                "loss": "SeqCeCriterion",
                "split_names": ["train", "dev", "test"],
                "task_type": "SequenceLabeling",
            }
        super(POSTaskConfig, self).__init__(**kwargs)
        self.labels = kwargs.pop(
            "labels",
            [
                ",",
                "\\",
                ":",
                ".",
                "''",
                '"',
                "(",
                ")",
                "$",
                "CC",
                "CD",
                "DT",
                "EX",
                "FW",
                "IN",
                "JJ",
                "JJR",
                "JJS",
                "LS",
                "MD",
                "NN",
                "NNP",
                "NNPS",
                "NNS",
                "NN|SYM",
                "PDT",
                "POS",
                "PRP",
                "PRP$",
                "RB",
                "RBR",
                "RBS",
                "RP",
                "SYM",
                "TO",
                "UH",
                "VB",
                "VBD",
                "VBG",
                "VBN",
                "VBP",
                "VBZ",
                "WDT",
                "WP",
                "WP$",
                "WRB",
                "X",
                "CLS",
                "SEP",
            ],
        )
        self.split_names = kwargs.pop("split_names", ["train", "dev", "test"])


class CHUNKTaskConfig(TaskConfig):
    def __init__(self, kwargs: dict = {}):
        if not kwargs:
            kwargs = {
                "task_name": "chunk",
                "data_format": "Seqence",
                "encoder_type": "BERT",
                "dropout_p": 0.1,
                "enable_san": False,
                "labels": [
                    "B-ADJP",
                    "B-ADVP",
                    "B-CONJP",
                    "B-INTJ",
                    "B-LST",
                    "B-NP",
                    "B-PP",
                    "B-PRT",
                    "B-SBAR",
                    "B-VP",
                    "I-ADJP",
                    "I-ADVP",
                    "I-CONJP",
                    "I-INTJ",
                    "I-LST",
                    "I-NP",
                    "I-PP",
                    "I-SBAR",
                    "I-VP",
                    "O",
                    "X",
                    "CLS",
                    "SEP",
                ],
                "metric_meta": ["SeqEval"],
                "n_class": 23,
                "loss": "SeqCeCriterion",
                "split_names": ["train", "dev", "test"],
                "task_type": "SequenceLabeling",
            }
        super(CHUNKTaskConfig, self).__init__(**kwargs)
        self.labels = kwargs.pop(
            "labels",
            [
                "B-ADJP",
                "B-ADVP",
                "B-CONJP",
                "B-INTJ",
                "B-LST",
                "B-NP",
                "B-PP",
                "B-PRT",
                "B-SBAR",
                "B-VP",
                "I-ADJP",
                "I-ADVP",
                "I-CONJP",
                "I-INTJ",
                "I-LST",
                "I-NP",
                "I-PP",
                "I-SBAR",
                "I-VP",
                "O",
                "X",
                "CLS",
                "SEP",
            ],
        )
        self.split_names = kwargs.pop("split_names", ["train", "dev", "test"])


class SQUADTaskConfig(TaskConfig):
    def __init__(self, kwargs: dict = {}):
        if not kwargs:
            kwargs = {
                "task_name": "squad",
                "data_format": "MRC",
                "encoder_type": "BERT",
                "dropout_p": 0.1,
                "enable_san": False,
                "metric_meta": ["EmF1"],
                "n_class": 2,
                "task_type": "Span",
                "loss": "SpanCeCriterion",
                "split_names": ["train", "dev"],
            }
        super(SQUADTaskConfig, self).__init__(**kwargs)
        self.split_names = kwargs.pop("split_names", ["train", "dev"])
        self.dropout_p = kwargs.pop("dropout_p", 0.1)


# Map of supported tasks
SUPPORTED_TASKS_MAP = {
    "cola": COLATaskConfig,
    "mnli": MNLITaskConfig,
    "mrpc": MRPCTaskConfig,
    "qnli": QNLITaskConfig,
    "qqp": QQPTaskConfig,
    "rte": RTETaskConfig,
    "scitail": SCITAILTaskConfig,
    "snli": SNLITaskConfig,
    "sst": SSTTaskConfig,
    "stsb": STSBTaskConfig,
    "wnli": WNLITaskConfig,
    "ner": NERTaskConfig,
    "pos": POSTaskConfig,
    "chunk": CHUNKTaskConfig,
    "squad": SQUADTaskConfig,
    "squad-v2": SQUADTaskConfig,
}


class MTDNNTaskConfig:
    supported_tasks_map = SUPPORTED_TASKS_MAP

    def from_dict(self, task_name: str, opts: dict = {}):
        """ Create Task configuration from dictionary of configuration """
        assert opts, "Configuration dictionary cannot be empty"
        task = self.supported_tasks_map[task_name]
        opts.update({"task_name": f"{task_name}"})
        return task(kwargs=opts)

    def get_supported_tasks(self) -> list:
        """Return list of supported tasks

        Returns:
            list -- Supported list of tasks
        """
        return self.supported_tasks_map.keys()

    # TODO - Cleanup
    # def get_configured_task(self) -> TaskConfig:
    #     """Get a Task Configuration

    #     Returns:
    #         TaskConfig -- TaskConfig Object to be configured
    #     """
    #     return self._config

    # def get_task_name(self) -> str:
    #     """Get the configured task name

    #     Returns:
    #         str -- TaskConfig string name to be configured
    #     """
    #     return self._task_name

    # def __repr__(self):
    #     return str(self.to_json_string())

    # def to_json_string(self):
    #     """Serializes this instance to a JSON string."""
    #     return json.dumps(self._kwargs, indent=2, sort_keys=True) + "\n"

    # def to_json_file(self, json_file_path):
    #     """ Save this instance to a json file."""
    #     with open(json_file_path, "w", encoding="utf-8") as fh:
    #         fh.write(self.to_json_string())

    # def save_config_file(self, save_directory: str = "/"):
    #     """ Save a configuration object to the directory `save_directory`"""
    #     assert os.path.isdir(
    #         save_directory
    #     ), "Saving path should be accessible to save the configuration file"

    #     # If we save using the predefined names, we can load using `from_pretrained`
    #     output_config_file = os.path.join(save_directory, self._task_name)

    #     self.to_json_file(output_config_file)
    #     logger.info("Configuration saved in {}".format(output_config_file))


class MTDNNTaskDefs:
    """Definition of single or multiple tasks to train. Can take a single task name or a definition yaml or json file
        
        Arguments:
            task_dict_or_def_file {str or dict} -- Task dictionary or definition file (yaml or json)  
            Example:

            JSON:
            {
                "cola": {
                    "data_format": "PremiseOnly",
                    "encoder_type": "BERT",
                    "dropout_p": 0.05,
                    "enable_san": false,
                    "metric_meta": [
                        "ACC",
                        "MCC"
                    ],
                    "loss": "CeCriterion",
                    "kd_loss": "MseCriterion",
                    "n_class": 2,
                    "task_type": "Classification"
                }
                ...
            }
            or 
            
            Python dict:
                { 
                    "cola": {
                        "data_format": "PremiseOnly",
                        "encoder_type": "BERT",
                        "dropout_p": 0.05,
                        "enable_san": False,
                        "metric_meta": [
                            "ACC",
                            "MCC"
                        ],
                        "loss": "CeCriterion",
                        "kd_loss": "MseCriterion",
                        "n_class": 2,
                        "task_type": "Classification"
                    }
                ...
            }

        """

    def __init__(self, task_dict_or_file: Union[str, dict]):

        assert task_dict_or_file, "Please pass in a task dict or definition file in yaml or json"
        self._task_def_dic = {}
        self._configured_tasks = []  # list of configured tasks
        if isinstance(task_dict_or_file, dict):
            self._task_def_dic = task_dict_or_file
        elif isinstance(task_dict_or_file, str):
            assert os.path.exists(task_dict_or_file), "Task definition file does not exist"
            assert os.path.isfile(task_dict_or_file), "Task definition must be a file"

            task_def_filepath, ext = os.path.splitext(task_dict_or_file)
            ext = ext[1:].lower()
            assert ext in ["json", "yml", "yaml",], "Definition file must be in JSON or YAML format"

            self._task_def_dic = (
                yaml.safe_load(open(task_dict_or_file))
                if ext in ["yaml", "yml"]
                else json.load(open(task_dict_or_file))
            )

        global_map = {}
        n_class_map = {}
        data_type_map = {}
        task_type_map = {}
        metric_meta_map = {}
        enable_san_map = {}
        dropout_p_map = {}
        encoderType_map = {}
        loss_map = {}
        kd_loss_map = {}

        # Create an instance of task creator singleton
        task_creator = MTDNNTaskConfig()

        uniq_encoderType = set()
        for name, params in self._task_def_dic.items():
            assert "_" not in name, f"task name should not contain '_', current task name: {name}"

            # Create a singleton to create tasks
            task = task_creator.from_dict(task_name=name, opts=params)

            n_class_map[name] = task.n_class
            data_type_map[name] = DataFormat[task.data_format]
            task_type_map[name] = TaskType[task.task_type]
            metric_meta_map[name] = tuple(Metric[metric_name] for metric_name in task.metric_meta)
            enable_san_map[name] = task.enable_san
            uniq_encoderType.add(EncoderModelType[task.encoder_type])

            if hasattr(task, "labels"):
                labels = task.labels
                label_mapper = Vocabulary(True)
                for label in labels:
                    label_mapper.add(label)
                global_map[name] = label_mapper

            # dropout
            if hasattr(task, "dropout_p"):
                dropout_p_map[name] = task.dropout_p

            # loss map
            if hasattr(task, "loss"):
                t_loss = task.loss
                loss_crt = LossCriterion[t_loss]
                loss_map[name] = loss_crt
            else:
                loss_map[name] = None

            if hasattr(task, "kd_loss"):
                t_loss = task.kd_loss
                loss_crt = LossCriterion[t_loss]
                kd_loss_map[name] = loss_crt
            else:
                kd_loss_map[name] = None

            # Track configured tasks for downstream
            self._configured_tasks.append(task.to_dict())

        logger.info(
            f"Configured task definitions - {[obj['task_name'] for obj in self.get_configured_tasks()]}"
        )

        assert len(uniq_encoderType) == 1, "The shared encoder has to be the same."
        self.global_map = global_map
        self.n_class_map = n_class_map
        self.data_type_map = data_type_map
        self.task_type_map = task_type_map
        self.metric_meta_map = metric_meta_map
        self.enable_san_map = enable_san_map
        self.dropout_p_map = dropout_p_map
        self.encoderType = uniq_encoderType.pop()
        self.loss_map = loss_map
        self.kd_loss_map = kd_loss_map

    def get_configured_tasks(self) -> list:
        """Returns a list of configured tasks by TaskDefs class from the input configuration file
        
        Returns:
            list -- List of configured task classes
        """
        return self._configured_tasks
