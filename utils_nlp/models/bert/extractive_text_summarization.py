# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
Wrapper for extractive summarization algorithm based on BERT, i.e. BERTSum. The code in this file reused some code  listed in https://github.com/nlpyang/BertSum/tree/master/src
"""

from pytorch_pretrained_bert import BertConfig

from bertsum.models.model_builder import Summarizer
from bertsum.models import model_builder, data_loader
from bertsum.others.logging import logger, init_logger
from bertsum.train import model_flags
from bertsum.models.trainer import build_trainer
from bertsum.prepro.data_builder import BertData
from bertsum.models.data_loader import DataIterator, Batch, Dataloader
from cached_property import cached_property
import torch
import random
from bertsum.prepro.data_builder import greedy_selection, combination_selection
import gc
from multiprocessing import Pool

from torch import nn
from torch.nn.parallel import DataParallel as DP


class Bunch(object):
    """ Class which convert a dictionary to an object """

    def __init__(self, adict):
        self.__dict__.update(adict)


default_parameters = {
    "accum_count": 1,
    "batch_size": 3000,
    "beta1": 0.9,
    "beta2": 0.999,
    "block_trigram": True,
    "decay_method": "noam",
    "dropout": 0.1,
    "encoder": "baseline",
    "ff_size": 512,
    "gpu_ranks": "0123",
    "heads": 4,
    "hidden_size": 128,
    "inter_layers": 2,
    "lr": 0.002,
    "max_grad_norm": 0,
    "max_nsents": 100,
    "max_src_ntokens": 200,
    "min_nsents": 3,
    "min_src_ntokens": 10,
    "optim": "adam",
    "oracle_mode": "combination",
    "param_init": 0.0,
    "param_init_glorot": True,
    "recall_eval": False,
    "report_every": 50,
    "report_rouge": True,
    "rnn_size": 512,
    "save_checkpoint_steps": 500,
    "seed": 666,
    "temp_dir": "./temp",
    "test_all": False,
    "test_from": "",
    "train_from": "",
    "use_interval": True,
    "visible_gpus": "0",
    "warmup_steps": 10000,
    "world_size": 1,
}

default_preprocessing_parameters = {
    "max_nsents": 100,
    "max_src_ntokens": 200,
    "min_nsents": 3,
    "min_src_ntokens": 10,
    "use_interval": True,
}


def bertsum_formatting(n_cpus, bertdata, oracle_mode, jobs, output_file):
    """
    Function to preprocess data for BERTSum algorithm.

    Args:
        n_cpus (int): number of cpus used for preprocessing in parallel
        bertdata (BertData): object which loads the pretrained BERT tokenizer to preprocess data.
        oracle_mode (string): name of the algorithm to select sentences in the source as labeled data correposonding to the target.  Options are "combination" and "greedy".
        jobs (list of dictionaries): list of dictionaries with "src" and "tgt" fields. Both fields should be filled with list of list of tokens/words.
        output_file (string): name of the file to save the processed data.
    """

    params = []
    for i in jobs:
        params.append((oracle_mode, bertdata, i))
    pool = Pool(n_cpus)
    bert_data = pool.map(modified_format_to_bert, params, int(len(params) / n_cpus))
    pool.close()
    pool.join()
    filtered_bert_data = []
    for i in bert_data:
        if i is not None:
            filtered_bert_data.append(i)
    torch.save(filtered_bert_data, output_file)


def modified_format_to_bert(param):
    """
    Helper function to preprocess data for BERTSum algorithm.

    Args: 
        param (Tuple): params are tuple of (string, BertData object, and dictionary). The first string specifies the oracle mode. The last dictionary should contain src" and "tgt" fields withc each filled with list of list of tokens/words.

    Returns:
        Dictionary: it has "src", "lables", "segs", "clss", "src_txt" and "tgt_txt" field.

    """

    oracle_mode, bert, data = param
    # return data
    source, tgt = data["src"], data["tgt"]
    if oracle_mode == "greedy":
        oracle_ids = greedy_selection(source, tgt, 3)
    elif oracle_mode == "combination":
        oracle_ids = combination_selection(source, tgt, 3)
    b_data = bert.preprocess(source, tgt, oracle_ids)
    if b_data is None:
        return None
    indexed_tokens, labels, segments_ids, cls_ids, src_txt, tgt_txt = b_data
    b_data_dict = {
        "src": indexed_tokens,
        "labels": labels,
        "segs": segments_ids,
        "clss": cls_ids,
        "src_txt": src_txt,
        "tgt_txt": tgt_txt,
    }
    return b_data_dict
    gc.collect()


def get_data_iter(dataset, is_test=False, batch_size=3000):
    """
    Function to get data iterator over a list of data objects.

    Args:
        dataset (list of objects): a list of data objects.
        is_test (bool): it specifies whether the data objects are labeled data.
        batch_size (int): number of tokens per batch.
        
    Returns:
        DataIterator

    """
    args = Bunch({})
    args.use_interval = True
    args.batch_size = batch_size
    test_data_iter = None
    test_data_iter = DataIterator(
        args, dataset, args.batch_size, "cuda", is_test=is_test, shuffle=False, sort=False
    )
    return test_data_iter


class BertSumExtractiveSummarizer:
    """ Wrapper class for BERT-based Extractive Summarization, i.e. BertSum"""

    def __init__(
        self,
        language="english",
        encoder="baseline",
        model_path="./models/baseline",
        log_file="./logs/baseline",
        temp_dir="./temp",
        bert_config_path="./bert_config_uncased_base.json",
        gpu_ranks="0",
    ):
        """Initializes the wrapper and the underlying pretrained model.
        Args:
            language (Language, optional): The pretrained model's language.
                                           Defaults to Language.ENGLISH.
            encoder (string, optional): the algorithm used for the Summarization layers. 
                                        Options are: baseline, transformer, rnn, classifier
            model_path (string, optional): path to save the checkpoints of the model for each training session
            log_files (string, optional): path to save the running logs for each session.
            temp_dir (string, optional): Location of BERT's cache directory.
                Defaults to ".".
            bert_config_path (string, optional): path of the config file for the BERT model
            gpu_ranks (string, optional): string with each character the string value of each GPU devices ID that can be used. Defaults to "0".
        """

        def __map_gpu_ranks(gpu_ranks):
            gpu_ranks_list = gpu_ranks.split(",")
            print(gpu_ranks_list)
            gpu_ranks_map = {}
            for i, rank in enumerate(gpu_ranks_list):
                gpu_ranks_map[int(rank)] = i
            return gpu_ranks_map

        # copy all the arguments from the input argument
        self.args = Bunch(default_parameters)
        self.args.seed = 42
        self.args.encoder = encoder
        self.args.model_path = model_path
        self.args.log_file = log_file
        self.args.temp_dir = temp_dir
        self.args.bert_config_path = bert_config_path

        self.args.gpu_ranks = gpu_ranks
        self.args.gpu_ranks_map = __map_gpu_ranks(self.args.gpu_ranks)
        self.args.world_size = len(self.args.gpu_ranks_map.keys())

        self.has_cuda = self.cuda
        init_logger(self.args.log_file)
        torch.manual_seed(self.args.seed)
        random.seed(self.args.seed)
        torch.backends.cudnn.deterministic = True
        # placeholder for the model
        self.model = None

    @cached_property
    def cuda(self):
        """ cache the output of torch.cuda.is_available() """

        self.has_cuda = torch.cuda.is_available()
        return self.has_cuda

    def fit(
        self,
        device_id,
        train_file_list,
        train_steps=5000,
        train_from="",
        batch_size=3000,
        warmup_proportion=0.2,
        decay_method="noam",
        lr=0.002,
        accum_count=2,
    ):
        """
        Train a summarization model with specified training data files.

        Args:
            device_id (string): GPU Device ID to be used.
            train_file_list (string): files used for training a model.
            train_steps (int, optional): number of times that the model parameters get updated. The number of data items for each model parameters update is the number of data items in a batch times times the accumulation counts (accum_count). Defaults to 5e5.
            train_from (string, optional): the path of saved checkpoints from which the model starts to train. Defaults to empty string.
            batch_size (int, options): maximum number of tokens in each batch.
            warmup_propotion (float, optional): Proportion of training to
                perform linear learning rate warmup for. E.g., 0.1 = 10% of
                training. Defaults to 0.2.
            decay_method (string, optional): learning rate decrease method. Default to 'noam'.
            lr (float, optional): Learning rate of the Adam optimizer. Defaults to 2e-3.
            accu_count (int, optional): number of batches waited until an update of the model paraeters happens. Defaults to 2.
        """

        if self.args.gpu_ranks_map[device_id] != 0:
            logger.disabled = True
        if device_id not in list(self.args.gpu_ranks_map.keys()):
            raise Exception("need to use device id that's in the gpu ranks")
        device = None
        if device_id >= 0:
            # torch.cuda.set_device(device_id)
            torch.cuda.manual_seed(self.args.seed)
            device = torch.device("cuda:{}".format(device_id))
            self.device = device

        self.args.decay_method = decay_method
        self.args.lr = lr
        self.args.train_from = train_from
        self.args.batch_size = batch_size
        self.args.warmup_steps = int(warmup_proportion * train_steps)
        self.args.accum_count = accum_count
        print(self.args.__dict__)

        self.model = Summarizer(self.args, device, load_pretrained_bert=True)

        self.model.to(device)
        self.model = DP(self.model, device_ids=[device])

        if train_from != "":
            checkpoint = torch.load(train_from, map_location=lambda storage, loc: storage)
            opt = vars(checkpoint["opt"])
            for k in opt.keys():
                if k in model_flags:
                    setattr(self.args, k, opt[k])
            self.model.load_cp(checkpoint)
            optim = model_builder.build_optim(self.args, self.model, checkpoint)
        else:
            optim = model_builder.build_optim(self.args, self.model, None)

        def get_dataset(file_list):
            random.shuffle(file_list)
            for file in file_list:
                yield torch.load(file)

        def train_iter_fct():
            return data_loader.Dataloader(
                self.args,
                get_dataset(train_file_list),
                batch_size,
                device,
                shuffle=True,
                is_test=True,
            )

        trainer = build_trainer(self.args, device_id, self.model, optim)
        trainer.train(train_iter_fct, train_steps)

    def predict(self, device_id, data_iter, sentence_seperator="", test_from="", cal_lead=False):
        """
        Predict the summarization for the input data iterator.

        Args:
            device_id (string): GPU Device ID to be used.
            data_iter (DataIterator): data iterator over the dataset to be predicted
            sentence_seperator (string, optional): strings to be inserted between sentences in the prediction per data item. Defaults to empty string.
            test_from(string, optional): the path of saved checkpoints used for prediction. 
            cal_lead (boolean, optional): wheather use the first three sentences as the prediction.
        """

        device = None
        if device_id >= 0:
            torch.cuda.manual_seed(self.args.seed)
            device = torch.device("cuda:{}".format(device_id))

        if self.model is None and test_from == "":
            raise Exception("Need to train or specify the model for testing")
        if test_from != "":
            checkpoint = torch.load(test_from, map_location=lambda storage, loc: storage)
            opt = vars(checkpoint["opt"])
            for k in opt.keys():
                if k in model_flags:
                    setattr(self.args, k, opt[k])

            config = BertConfig.from_json_file(self.args.bert_config_path)
            self.model = Summarizer(
                self.args, device, load_pretrained_bert=False, bert_config=config
            )

            class WrappedModel(nn.Module):
                def __init__(self, module):
                    super(WrappedModel, self).__init__()
                    self.module = module

                def forward(self, x):
                    return self.module(x)

            model = WrappedModel(self.model)
            # self.model.load_cp(checkpoint)
            model.load_state_dict(checkpoint["model"])
            self.model = model.module
        else:
            self.model.eval()
        self.model.eval()

        self.model.to(device)
        self.model = DP(self.model, device_ids=[device])

        trainer = build_trainer(self.args, device_id, self.model, None)
        return trainer.predict(data_iter, sentence_seperator, cal_lead)
