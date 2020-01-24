# Copyright (c) Microsoft. All rights reserved.
from enum import Enum

from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr, spearmanr
from seqeval.metrics import classification_report
from utils_nlp.models.mtdnn.common.squad_eval import evaluate_func


def compute_acc(predicts, labels):
    return 100.0 * accuracy_score(labels, predicts)


def compute_f1(predicts, labels):
    return 100.0 * f1_score(labels, predicts)


def compute_mcc(predicts, labels):
    return 100.0 * matthews_corrcoef(labels, predicts)


def compute_pearson(predicts, labels):
    pcof = pearsonr(labels, predicts)[0]
    return 100.0 * pcof


def compute_spearman(predicts, labels):
    scof = spearmanr(labels, predicts)[0]
    return 100.0 * scof


def compute_auc(predicts, labels):
    auc = roc_auc_score(labels, predicts)
    return 100.0 * auc


def compute_seqacc(predicts, labels, label_mapper):
    y_true, y_pred = [], []

    def trim(predict, label):
        temp_1 = []
        temp_2 = []
        for j, m in enumerate(predict):
            if j == 0:
                continue
            if label_mapper[label[j]] != "X":
                temp_1.append(label_mapper[label[j]])
                temp_2.append(label_mapper[m])
        temp_1.pop()
        temp_2.pop()
        y_true.append(temp_1)
        y_pred.append(temp_2)

    for predict, label in zip(predicts, labels):
        trim(predict, label)
    report = classification_report(y_true, y_pred, digits=4)
    return report


def compute_emf1(predicts, labels):
    return evaluate_func(labels, predicts)


class Metric(Enum):
    ACC = 0
    F1 = 1
    MCC = 2
    Pearson = 3
    Spearman = 4
    AUC = 5
    SeqEval = 7
    EmF1 = 8


METRIC_FUNC = {
    Metric.ACC: compute_acc,
    Metric.F1: compute_f1,
    Metric.MCC: compute_mcc,
    Metric.Pearson: compute_pearson,
    Metric.Spearman: compute_spearman,
    Metric.AUC: compute_auc,
    Metric.SeqEval: compute_seqacc,
    Metric.EmF1: compute_emf1,
}


def calc_metrics(metric_meta, golds, predictions, scores, label_mapper=None):
    """Label Mapper is used for NER/POS etc. 
    TODO: a better refactor, by xiaodl
    """
    metrics = {}
    for mm in metric_meta:
        metric_name = mm.name
        metric_func = METRIC_FUNC[mm]
        if mm in (Metric.ACC, Metric.F1, Metric.MCC):
            metric = metric_func(predictions, golds)
        elif mm == Metric.SeqEval:
            metric = metric_func(predictions, golds, label_mapper)
        elif mm == Metric.EmF1:
            metric = metric_func(predictions, golds)
        else:
            if mm == Metric.AUC:
                assert len(scores) == 2 * len(
                    golds
                ), "AUC is only valid for binary classification problem"
                scores = scores[1::2]
            metric = metric_func(scores, golds)
        metrics[metric_name] = metric
    return metrics
