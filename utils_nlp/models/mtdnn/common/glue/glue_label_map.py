# Copyright (c) Microsoft. All rights reserved.

from data_utils.vocab import Vocabulary
from data_utils.metrics import compute_acc, compute_f1, compute_mcc, compute_pearson, compute_spearman

# scitail
ScitailLabelMapper = Vocabulary(True)
ScitailLabelMapper.add('neutral')
ScitailLabelMapper.add('entails')

# label map
SNLI_LabelMapper = Vocabulary(True)
SNLI_LabelMapper.add('contradiction')
SNLI_LabelMapper.add('neutral')
SNLI_LabelMapper.add('entailment')

# qnli
QNLILabelMapper = Vocabulary(True)
QNLILabelMapper.add('not_entailment')
QNLILabelMapper.add('entailment')

GLOBAL_MAP = {
    'scitail': ScitailLabelMapper,
    'mnli': SNLI_LabelMapper,
    'snli': SNLI_LabelMapper,
    'qnli': QNLILabelMapper,
    'qnnli': QNLILabelMapper,
    'rte': QNLILabelMapper,
    'diag': SNLI_LabelMapper,
}

# number of class
DATA_META = {
    'mnli': 3,
    'snli': 3,
    'scitail': 2,
    'qqp': 2,
    'qnli': 2,
    'qnnli': 1,
    'wnli': 2,
    'rte': 2,
    'mrpc': 2,
    'diag': 3,
    'sst': 2,
    'stsb': 1,
    'cola': 2,
}

DATA_TYPE = {
    'mnli': 0,
    'snli': 0,
    'scitail': 0,
    'qqp': 0,
    'qnli': 0,
    'qnnli': 0,
    'wnli': 0,
    'rte': 0,
    'mrpc': 0,
    'diag': 0,
    'sst': 1,
    'stsb': 0,
    'cola': 1,
}

DATA_SWAP = {
    'mnli': 0,
    'snli': 0,
    'scitail': 0,
    'qqp': 1,
    'qnli': 0,
    'qnnli': 0,
    'wnli': 0,
    'rte': 0,
    'mrpc': 0,
    'diag': 0,
    'sst': 0,
    'stsb': 0,
    'cola': 0,
}

# classification/regression
TASK_TYPE = {
    'mnli': 0,
    'snli': 0,
    'scitail': 0,
    'qqp': 0,
    'qnli': 0,
    'qnnli': 0,
    'wnli': 0,
    'rte': 0,
    'mrpc': 0,
    'diag': 0,
    'sst': 0,
    'stsb': 1,
    'cola': 0,
}

METRIC_META = {
    'mnli': [0],
    'snli': [0],
    'scitail': [0],
    'qqp': [0, 1],
    'qnli': [0],
    'qnnli': [0],
    'wnli': [0],
    'rte': [0],
    'mrpc': [0, 1],
    'diag': [0],
    'sst': [0],
    'stsb': [3, 4],
    'cola': [0, 2],
}

METRIC_NAME = {
    0: 'ACC',
    1: 'F1',
    2: 'MCC',
    3: 'Pearson',
    4: 'Spearman',
}

METRIC_FUNC = {
    0: compute_acc,
    1: compute_f1,
    2: compute_mcc,
    3: compute_pearson,
    4: compute_spearman,
}

SAN_META = {
    'mnli': 1,
    'snli': 1,
    'scitail': 1,
    'qqp': 1,
    'qnli': 1,
    'qnnli': 1,
    'wnli': 1,
    'rte': 1,
    'mrpc': 1,
    'diag': 0,
    'sst': 0,
    'stsb': 0,
    'cola': 0,
}


def generate_decoder_opt(task, max_opt):
    assert task in SAN_META
    opt_v = 0
    if SAN_META[task] and max_opt < 3:
        opt_v = max_opt
    return opt_v


from enum import Enum


class TaskType(Enum):
    Classification = 0
    Regression = 1
    Ranking = 2
