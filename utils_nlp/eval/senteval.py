# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Utilities for evaluating sentence embeddings."""


class SentEvalConfig:
    """Object to store static properties of senteval experiments

    Attributes:
        model_params (dict): model parameters that stay consistent across all runs
        senteval_params (dict): senteval parameters that stay consistent across all runs

    """

    def __init__(self, model_params, senteval_params):
        """Summary

        Args:
            model_params (dict): model parameters that stay consistent across all runs
            senteval_params (dict): senteval parameters that stay consistent across all runs
        """
        self.model_params = model_params
        self.senteval_params = senteval_params

    @property
    def model_params(self):
        return self._model_params

    @model_params.setter
    def model_params(self, model_params):
        self._model_params = model_params

    def append_senteval_params(self, params):
        """Util to append any params to senteval_params after initialization"""
        self.senteval_params = dict(self.senteval_params, **params)

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
                a = "classifier" in self.senteval_params
                if not a:
                    raise ValueError("Include param['classifier'] to run task {}".format(t))
                else:
                    b = (
                        set("nhid", "optim", "batch_size", "tenacity", "epoch_size")
                        in self.senteval_params["classifier"].keys()
                    )
                    if not b:
                        raise ValueError(
                            "Include nhid, optim, batch_size, tenacity, and epoch_size params to "
                            "run task {}".format(t)
                        )
            except ValueError as ve:
                print(ve)
