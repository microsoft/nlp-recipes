import os
import sys
import pandas as pd


class SentEvalRunner:
    def __init__(self, path_to_senteval="."):
        """Wrapper class interfacing with the original implementation of SentEval
        
        Args:
            path_to_senteval (str, optional): Path to the SentEval source code.
        """
        self.path_to_senteval = path_to_senteval
        self.transfer_data_path = os.path.join(self.path_to_senteval, "data")
        self.params_senteval = {"task_path": self.transfer_data_path}

    def set_transfer_data_path(self, path):
        """Set the datapath that contains the datasets for the SentEval transfer tasks
        
        Args:
            path (str): Path to transfer task datasets
        """
        self.transfer_data_path = path
        self.params_senteval["task_path"] = path

    def set_transfer_tasks(self, task_list):
        """Set the transfer tasks to use for evaluation
        
        Args:
            task_list (list(str)): List of downstream transfer tasks
        """
        self.transfer_tasks = task_list

    def set_model(self, model):
        """Set the model to evaluate"""
        self.params_senteval["model"] = model

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

    def run(self, batcher_func, prepare_func):
        """Run the SentEval engine on the model on the transfer tasks
        
        Args:
            batcher_func (function): Function required by SentEval that transforms a batch of text sentences into 
                                     sentence embeddings
            prepare_func (function): Function that sees the whole dataset of each task and can thus construct the word 
                                     vocabulary, the dictionary of word vectors, etc
        
        Returns:
            dict: Dictionary of results
        """
        sys.path.insert(0, self.path_to_senteval)
        import senteval

        se = senteval.engine.SE(
            self.params_senteval, batcher_func, prepare_func
        )

        return se.eval(self.transfer_tasks)

    def log_mean(self, results, selected_metrics=[], round_decimals=3):
        """Log the means of selected metrics of the transfer tasks
        
        Args:
            results (dict): Results from the SentEval evaluation engine
            selected_metrics (list(str), optional): List of metric names
            round_decimals (int, optional): Number of decimal digits to round to; defaults to 3
        
        Returns:
            pd.DataFrame table of formatted results
        """
        data = []
        for task in self.transfer_tasks:
            if "all" in results[task]:
                row = [
                    results[task]["all"][metric]["mean"]
                    for metric in selected_metrics
                ]
            else:
                row = [results[task][metric] for metric in selected_metrics]
            data.append(row)
        table = pd.DataFrame(
            data=data, columns=selected_metrics, index=self.transfer_tasks
        )
        return table.round(round_decimals)
