# coding=utf-8
# Copyright (c) Microsoft. All rights reserved.
import os
import torch
import subprocess


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

