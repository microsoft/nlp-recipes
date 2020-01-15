# coding=utf-8
# Copyright (c) Microsoft. All rights reserved.
from copy import deepcopy
from functools import wraps

import torch
from torch.nn import Parameter


class EMA:
    def __init__(self, gamma, model):
        super(EMA, self).__init__()
        self.gamma = gamma
        self.shadow = {}
        self.model = model
        self.setup()

    def setup(self):
        for name, para in self.model.named_parameters():
            if para.requires_grad:
                self.shadow[name] = para.clone()

    def cuda(self):
        for k, v in self.shadow.items():
            self.shadow[k] = v.cuda()

    def update(self):
        for name, para in self.model.named_parameters():
            if para.requires_grad:
                self.shadow[name] = (1.0 - self.gamma) * para + self.gamma * self.shadow[name]

    def swap_parameters(self):
        for name, para in self.model.named_parameters():
            if para.requires_grad:
                temp_data = para.data
                para.data = self.shadow[name].data
                self.shadow[name].data = temp_data

    def state_dict(self):
        return self.shadow


# Adapted from
# https://github.com/pytorch/pytorch/blob/master/torch/nn/utils/weight_norm.py
# and https://github.com/salesforce/awd-lstm-lm/blob/master/weight_drop.py
def _norm(p, dim):
    """Computes the norm over all dimensions except dim"""
    if dim is None:
        return p.norm()
    elif dim == 0:
        output_size = (p.size(0),) + (1,) * (p.dim() - 1)
        return p.contiguous().view(p.size(0), -1).norm(dim=1).view(*output_size)
    elif dim == p.dim() - 1:
        output_size = (1,) * (p.dim() - 1) + (p.size(-1),)
        return p.contiguous().view(-1, p.size(-1)).norm(dim=0).view(*output_size)
    else:
        return _norm(p.transpose(0, dim), 0).transpose(0, dim)


def _dummy(*args, **kwargs):
    # We need to replace flatten_parameters with a nothing function
    return


class WeightNorm(torch.nn.Module):
    def __init__(self, weights, dim):
        super(WeightNorm, self).__init__()
        self.weights = weights
        self.dim = dim

    def compute_weight(self, module, name):
        g = getattr(module, name + "_g")
        v = getattr(module, name + "_v")
        return v * (g / _norm(v, self.dim))

    @staticmethod
    def apply(module, weights, dim):
        # Terrible temporary solution to an issue regarding compacting weights
        # re: CUDNN RNN
        if issubclass(type(module), torch.nn.RNNBase):
            module.flatten_parameters = _dummy
        if weights is None:  # do for all weight params
            weights = [w for w in module._parameters.keys() if "weight" in w]
        fn = WeightNorm(weights, dim)
        for name in weights:
            if hasattr(module, name):
                print("Applying weight norm to {} - {}".format(str(module), name))
                weight = getattr(module, name)
                del module._parameters[name]
                module.register_parameter(name + "_g", Parameter(_norm(weight, dim).data))
                module.register_parameter(name + "_v", Parameter(weight.data))
                setattr(module, name, fn.compute_weight(module, name))

        module.register_forward_pre_hook(fn)

        return fn

    def remove(self, module):
        for name in self.weights:
            weight = self.compute_weight(module)
            delattr(module, name)
            del module._parameters[name + "_g"]
            del module._parameters[name + "_v"]
            module.register_parameter(name, Parameter(weight.data))

    def __call__(self, module, inputs):
        for name in self.weights:
            setattr(module, name, self.compute_weight(module, name))


def weight_norm(module, weights=None, dim=0):
    WeightNorm.apply(module, weights, dim)
    return module
