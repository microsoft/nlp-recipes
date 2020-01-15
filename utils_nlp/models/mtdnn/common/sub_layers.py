# coding=utf-8
# Copyright (c) Microsoft. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

class LayerNorm(nn.Module):
    #ref: https://github.com/pytorch/pytorch/issues/1959
    #   :https://arxiv.org/pdf/1607.06450.pdf
    def __init__(self, hidden_size, eps=1e-4):
        super(LayerNorm, self).__init__()
        self.alpha = Parameter(torch.ones(1,1,hidden_size)) # gain g
        self.beta = Parameter(torch.zeros(1,1,hidden_size)) # bias b
        self.eps = eps

    def forward(self, x):
        """
        Args:
            :param x: batch * len * input_size

        Returns:
            normalized x
        """
        mu = torch.mean(x, 2, keepdim=True).expand_as(x)
        sigma = torch.std(x, 2, keepdim=True).expand_as(x)
        return (x - mu) / (sigma + self.eps) * self.alpha.expand_as(x) + self.beta.expand_as(x)
