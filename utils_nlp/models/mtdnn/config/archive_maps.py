# coding=utf-8
# Copyright (c) Microsoft. All rights reserved.

from task_config import (
    COLATaskConfig,
    MNLITaskConfig,
    MRPCTaskConfig,
    QNLITaskConfig,
    QQPTaskConfig,
    RTETaskConfig,
    SCITAILTaskConfig,
    SNLITaskConfig,
    SSTTaskConfig,
    STSBTaskConfig,
    WNLITaskConfig,
    NERTaskConfig,
    POSTaskConfig,
    CHUNKTaskConfig,
    SQUADTaskConfig,
)

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


PRETRAINED_MODEL_ARCHIVE_MAP = {
    "mtdnn-base-uncased": "https://mrc.blob.core.windows.net/mt-dnn-model/mt_dnn_base.pt",
    "mtdnn-large-uncased": "https://mrc.blob.core.windows.net/mt-dnn-model/mt_dnn_large.pt",
    "mtdnn-kd-large-cased": "https://mrc.blob.core.windows.net/mt-dnn-model/mt_dnn_kd_large_cased.pt",
}
