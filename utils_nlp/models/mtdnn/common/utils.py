# coding=utf-8
# Copyright (c) Microsoft. All rights reserved.
import json
import logging
import os
import subprocess
from logging import Logger

import torch


class MTDNNCommonUtils:
    @staticmethod
    def set_environment(seed, set_cuda=False):
        random.seed(seed)
        numpy.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available() and set_cuda:
            torch.cuda.manual_seed_all(seed)

    @staticmethod
    def patch_var(v, cuda=True):
        if cuda:
            v = v.cuda(non_blocking=True)
        return v

    @staticmethod
    def get_gpu_memory_map():
        result = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,nounits,noheader"],
            encoding="utf-8",
        )
        gpu_memory = [int(x) for x in result.strip().split("\n")]
        gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
        return gpu_memory_map

    @staticmethod
    def get_pip_env():
        result = subprocess.call(["pip", "freeze"])
        return result

    @staticmethod
    def load_pytorch_model(local_model_path: str = ""):
        state_dict = None
        assert os.path.exists(local_model_path), "Model File path doesn't exist"
        state_dict = torch.load(local_model_path)
        return state_dict

    @staticmethod
    def dump(path, data):
        with open(path, "w") as f:
            json.dump(data, f)

    @staticmethod
    def generate_decoder_opt(enable_san, max_opt):
        opt_v = 0
        if enable_san and max_opt < 3:
            opt_v = max_opt
        return opt_v

    @staticmethod
    def setup_logging(filename="run.log", mode="a") -> Logger:
        logger = logging.getLogger(__name__)
        log_file_handler = logging.FileHandler(filename="run.log", mode="a")
        log_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        log_file_handler.setFormatter(log_formatter)
        do_add_handler = True
        for handler in logger.handlers:
            if isinstance(handler, logging.FileHandler):
                do_add_handler = False
        if do_add_handler:
            logger.addHandler(log_file_handler)
        logger.setLevel(logging.DEBUG)
        return logger

    @staticmethod
    def create_directory_if_not_exists(dir_path: str):
        os.makedirs(dir_path, exist_ok=True)
