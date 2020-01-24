# coding=utf-8
# Copyright (c) Microsoft. All rights reserved.


class MTDNNDataPreprocess:
    def __init__(self, dataset_list: list = ["mnli"]):
        self.dataset_list = dataset_list

    def generate_decoder_opt(self, enable_san, max_opt):
        opt_v = 0
        if enable_san and max_opt < 3:
            opt_v = max_opt
        return opt_v
