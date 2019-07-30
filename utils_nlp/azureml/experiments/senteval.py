import os
import sys
import itertools
import pandas as pd
import pickle
from collections import OrderedDict
from copy import deepcopy
from azureml.core import Experiment, ScriptRunConfig

from utils_nlp.azureml.azureml_utils import (
    get_or_create_workspace,
    get_or_create_amlcompute,
)
from azureml.core.runconfig import RunConfiguration



class SentEvalRunner:
    def __init__(
        self,
        config_path,
        compute_name,
        datastore_prefix,
        experiment_name,
        senteval_config,
        experiment_params,
        src_dir=".",
        verbose=False,
    ):
        self.compute_name = compute_name
        self.senteval_config = senteval_config
        self.experiment_params = experiment_params
        self.src_dir = src_dir
        self.verbose = verbose

        self._prepare_workspace(config_path)
        self._prepare_experiment(experiment_name)
        self._prepare_datastore()
        self._upload_senteval(datastore_prefix)

    def _prepare_workspace(self, config_path):
        self.ws = get_or_create_workspace(config_path=config_path)

    def _prepare_experiment(self, experiment_name):
        self.experiment = Experiment(
            workspace=self.ws, name=experiment_name
        )

    def _prepare_compute_target(self, num_nodes):
        self.compute = get_or_create_amlcompute(
            workspace=self.ws,
            compute_name=self.compute_name,
            vm_size="STANDARD_NC6",
            max_nodes=num_nodes,
            idle_seconds_before_scaledown=300,
            verbose=False,
        )

    def _prepare_datastore(self):
        self.ds = self.ws.get_default_datastore()
        if self.verbose:
            print("Uploading to default datastore: {}".format(self.ds.name))

    def _upload_senteval(self, datastore_prefix):
        self.ds.upload(
            src_dir=self.senteval_config.path,
            target_path=os.path.join(datastore_prefix, "senteval"),
            overwrite=False,
            show_progress=self.verbose,
        )

    def run(self):
        parameter_groups = list(
            itertools.product(*list(self.experiment_params.values()))
        )
        self._prepare_compute_target(num_nodes=len(parameter_groups))

        rc = RunConfiguration(framework="PyTorch")
        rc.target = self.compute_name

        for i, p in enumerate(parameter_groups):
            exp_params = dict(zip(self.experiment_params.keys(), p))
            sc = deepcopy(self.senteval_config)
            sc.append_params(exp_params)
            for k, v in exp_params.items():
                print(k,v)
                setattr(sc.model, k, v)
            
            pickle.dump(sc, open(os.path.join(self.src_dir, "config{0:03d}.pkl".format(i)), "wb"))

            src = ScriptRunConfig(
                source_directory=self.src_dir,
                script="run.py",
                arguments=[
                    "--config",
                    os.path.join(self.src_dir, "config{0:03d}.pkl".format(i)),
                    "--output",
                    os.path.join(self.src_dir, "results{0:03d}.pkl".format(i)),
                ],
                run_config=rc,
            )

            self.experiment.submit(src)
