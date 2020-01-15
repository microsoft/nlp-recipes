# coding=utf-8
# Copyright (c) Microsoft. All rights reserved.
from data_utils.metrics import calc_metrics
from mt_dnn.batcher import Collater
from data_utils.task_def import TaskType
def eval_model(model, data, metric_meta, use_cuda=True, with_label=True, label_mapper=None, task_type=TaskType.Classification):
    if use_cuda:
        model.cuda()
    predictions = []
    golds = []
    scores = []
    ids = []
    metrics = {}
    for idx, (batch_info, batch_data) in enumerate(data):
        if idx % 100 == 0:
            print("predicting {}".format(idx))
        batch_info, batch_data = Collater.patch_data(use_cuda, batch_info, batch_data)
        score, pred, gold = model.predict(batch_info, batch_data)
        predictions.extend(pred)
        golds.extend(gold)
        scores.extend(score)
        ids.extend(batch_info['uids'])

    if task_type == TaskType.Span:
        from experiments.squad import squad_utils
        golds = squad_utils.merge_answers(ids, golds)
        predictions, scores = squad_utils.select_answers(ids, predictions, scores)
    if with_label:
        metrics = calc_metrics(metric_meta, golds, predictions, scores, label_mapper)
    return metrics, predictions, scores, golds, ids
