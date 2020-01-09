# coding=utf-8
# Copyright (c) Microsoft. All rights reserved.

# This script reuses some code from https://github.com/huggingface/transformers


""" Model configuration """

from __future__ import absolute_import, division, print_function, unicode_literals

import json
import logging
import os

logger = logging.getLogger(__name__)


class TaskConfig(object):
    """Base Class for Task Configurations

    Handles parameters that are common to all task configurations

    Arguments:
        object {[type]} -- [description]
    """

    def __init__(self, **kwargs: dict):
        """ Define a generic task configuration """
        logger.info("Inside Task Config Init")
        self.data_format = kwargs.pop("data_format", "PremiseOnly")
        self.encoder_type = kwargs.pop("encoder_type", "BERT")
        self.enable_san = kwargs.pop("enable_san", False)
        self.metric_meta = kwargs.pop("metric_meta", ["ACC"])
        self.n_class = kwargs.pop("n_class", 2)
        self.task_type = kwargs.pop("task_type", "Classification")


class COLATaskConfig(TaskConfig):
    def __init__(self, **kwargs: dict):
        super(COLATaskConfig, self).__init__(**kwargs)
        self.dropout_p = kwargs.pop("dropout_p", 0.05)


class MNLITaskConfig(TaskConfig):
    def __init__(self, **kwargs: dict):
        super(MNLITaskConfig, self).__init__(**kwargs)
        self.dropout_p = kwargs.pop("dropout_p", 0.3)
        self.split_names = kwargs.pop(
            "split_names",
            ["train", "matched_dev", "mismatched_dev", "matched_test", "mismatched_test"],
        )


class MRPCTaskConfig(TaskConfig):
    def __init__(self, **kwargs: dict):
        super(MRPCTaskConfig, self).__init__(**kwargs)


class QNLITaskConfig(TaskConfig):
    def __init__(self, **kwargs: dict):
        super(QNLITaskConfig, self).__init__(**kwargs)
        self.labels = kwargs.pop("labels", ["not_entailment", "entailment"])


class QQPTaskConfig(TaskConfig):
    def __init__(self, **kwargs: dict):
        super(QQPTaskConfig, self).__init__(**kwargs)


class RTETaskConfig(TaskConfig):
    def __init__(self, **kwargs: dict):
        super(RTETaskConfig, self).__init__(**kwargs)
        self.labels = kwargs.pop("labels", ["not_entailment", "entailment"])


class SCITAILTaskConfig(TaskConfig):
    def __init__(self, **kwargs: dict):
        super(SCITAILTaskConfig, self).__init__(**kwargs)
        self.labels = kwargs.pop("labels", ["neutral", "entails"])


class SNLITaskConfig(TaskConfig):
    def __init__(self, **kwargs: dict):
        super(SNLITaskConfig, self).__init__(**kwargs)
        self.labels = kwargs.pop("labels", ["contradiction", "neutral", "entailment"])


class SSTTaskConfig(TaskConfig):
    def __init__(self, **kwargs: dict):
        super(SSTTaskConfig, self).__init__(**kwargs)


class STSBTaskConfig(TaskConfig):
    def __init__(self, **kwargs: dict):
        super(STSBTaskConfig, self).__init__(**kwargs)


class WNLITaskConfig(TaskConfig):
    def __init__(self, **kwargs: dict):
        super(WNLITaskConfig, self).__init__(**kwargs)


class NERTaskConfig(TaskConfig):
    def __init__(self, **kwargs: dict):
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
    def __init__(self, **kwargs: dict):
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
    def __init__(self, **kwargs: dict):
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
    def __init__(self, **kwargs: dict):
        super(SQUADTaskConfig, self).__init__(**kwargs)
        self.split_names = kwargs.pop("split_names", ["train", "dev"])
        self.dropout_p = kwargs.pop("dropout_p", 0.05)


class MTDNNTaskConfig:
    def __init__(self, **kwargs: dict):
        self._kwargs = kwargs
        self._task_name = kwargs.pop("task_name", "")
        self._task_class = SUPPORTED_TASKS_MAP[self._task_name]
        self._config = self._task_class(kwargs=self._kwargs)

    def get_supported_tasks(self) -> list:
        """Return list of supported tasks

        Returns:
            list -- Supported list of tasks
        """
        return SUPPORTED_TASKS_MAP.keys()

    def get_configured_task(self) -> TaskConfig:
        """Get a Task Configuration

        Returns:
            TaskConfig -- TaskConfig Object to be configured
        """
        return self._config

    def get_task_name(self) -> str:
        """Get the configured task name
        
        Returns:
            str -- TaskConfig string name to be configured
        """
        return self._task_name

    def __repr__(self):
        return str(self.to_json_string())

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self._kwargs, indent=2, sort_keys=True) + "\n"

    def to_json_file(self, json_file_path):
        """ Save this instance to a json file."""
        with open(json_file_path, "w", encoding="utf-8") as fh:
            fh.write(self.to_json_string())

    def save_config_file(self, save_directory: str = "/"):
        """ Save a configuration object to the directory `save_directory`"""
        assert os.path.isdir(
            save_directory
        ), "Saving path should be accessible to save the configuration file"

        # If we save using the predefined names, we can load using `from_pretrained`
        output_config_file = os.path.join(save_directory, self._task_name)

        self.to_json_file(output_config_file)
        logger.info("Configuration saved in {}".format(output_config_file))

