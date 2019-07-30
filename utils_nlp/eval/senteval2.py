import os
import sys
import itertools
import pandas as pd
from collections import OrderedDict
from copy import deepcopy


class SentEvalConfig:
    def __init__(
        self,
        path=".",
        model=None,
        prepare_func=None,
        batcher_func=None,
        transfer_tasks=None,
    ):
        self.path = path
        self.params_senteval = {}
        self.transfer_data_path = os.path.join(self.path, "data")
        self.model = model
        self.prepare_func = prepare_func
        self.batcher_func = batcher_func
        self.transfer_tasks = transfer_tasks

    @property
    def path(self):
        return self._path

    @path.setter
    def path(self, path):
        self._path = path

    @property
    def transfer_data_path(self):
        return self._transfer_data_path

    @transfer_data_path.setter
    def transfer_data_path(self, transfer_data_path):
        self._transfer_data_path = transfer_data_path
        self.params_senteval["task_path"] = transfer_data_path

    @property
    def transfer_tasks(self):
        return self._transfer_tasks

    @transfer_tasks.setter
    def transfer_tasks(self, transfer_tasks):
        self._transfer_tasks = transfer_tasks

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model):
        self._model = model
        self.params_senteval["model"] = model

    @property
    def prepare_func(self):
        return self._prepare_func

    @prepare_func.setter
    def prepare_func(self, prepare_func):
        self._prepare_func = prepare_func

    @property
    def batcher_func(self):
        return self._batcher_func

    @batcher_func.setter
    def batcher_func(self, batcher_func):
        self._batcher_func = batcher_func

    def append_params(self, params):
        self.params_senteval = dict(self.params_senteval, **params)

        classifying_tasks = {
            "MR",
            "CR",
            "SUBJ",
            "MPQA",
            "SST2",
            "SST5",
            "TREC",
            "SICKEntailment",
            "SNLI",
            "MRPC",
        }

        if any(t in classifying_tasks for t in self.transfer_tasks):
            try:
                a = "classifier" in self.params_senteval
                if not a:
                    raise ValueError(
                        "Include param['classifier'] to run task {}".format(t)
                    )
                else:
                    b = (
                        set(
                            "nhid",
                            "optim",
                            "batch_size",
                            "tenacity",
                            "epoch_size",
                        )
                        in self.params_senteval["classifier"].keys()
                    )
                    if not b:
                        raise ValueError(
                            "Include nhid, optim, batch_size, tenacity, and epoch_size params to run task {}".format(
                                t
                            )
                        )
            except ValueError as ve:
                print(ve)


# class ExperimentRunner:
#     def __init__(self, senteval_config, experiment_parameters={}):
#         self.senteval_config = senteval_config
#         self.experiment_parameters = experiment_parameters

#     def run(self, senteval_config):
#         sys.path.insert(0, senteval_config.path_to_senteval)
#         import senteval

#         se = senteval.engine.SE(
#             senteval_config.params_senteval, senteval_config.batcher_func, senteval_config.prepare_func
#         )

#         return se.eval(senteval_config.transfer_tasks)

#     def run_all(self):
#         all_metrics = []

#         for p in list(
#             itertools.product(*list(self.experiment_parameters.values()))
#         ):
#             exp_params = dict(zip(self.experiment_parameters.keys(), p))
#             experiment = deepcopy(self.senteval_config)
#             experiment.append_params(exp_params)
#             for k,v in exp_params.items():
#                 setattr(experiment.model, k, v)

#             results = self.run(experiment)
#             eval_metrics = log_mean(
#                 results,
#                 experiment.transfer_tasks,
#                 selected_metrics=["pearson", "spearman"],
#             )
#             for k,v in exp_params.items():
#                 eval_metrics[k] = v
#             all_metrics.append(eval_metrics)

#         return pd.concat(all_metrics, ignore_index=True)


def log_mean(
    results, transfer_tasks=[], selected_metrics=[], round_decimals=3
):
    """Log the means of selected metrics of the transfer tasks
    
    Args:
        results (dict): Results from the SentEval evaluation engine
        selected_metrics (list(str), optional): List of metric names
        round_decimals (int, optional): Number of decimal digits to round to; defaults to 3
    
    Returns:
        pd.DataFrame table of formatted results
    """
    data = []
    for task in transfer_tasks:
        if "all" in results[task]:
            row = [
                results[task]["all"][metric]["mean"]
                for metric in selected_metrics
            ]
        else:
            row = [results[task][metric] for metric in selected_metrics]
        data.append(row)
    table = pd.DataFrame(
        data=data, columns=selected_metrics, index=transfer_tasks
    )
    return table.round(round_decimals)
