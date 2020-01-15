# coding=utf-8
# Copyright (c) Microsoft. All rights reserved.
import math
import torch
from torch.optim import Optimizer
from torch.nn.utils import clip_grad_norm_
from pytorch_pretrained_bert.optimization import warmup_constant, warmup_cosine, warmup_linear


def warmup_linear_xdl(x, warmup=0.002):
    if x < warmup:
        return x / warmup
    return (1.0 - x) / (1.0 - warmup)


def schedule_func(sch):
    try:
        f = eval(sch)
    except:
        f = warmup_linear
    return f


class Adamax(Optimizer):
    """Implements BERT version of Adam algorithm with weight decay fix (and no ).
    Params:
        lr: learning rate
        warmup: portion of t_total for the warmup, -1  means no warmup. Default: -1
        t_total: total number of training steps for the learning
            rate schedule, -1  means constant learning rate. Default: -1
        schedule: schedule to use for the warmup (see above). Default: 'warmup_linear'
        b1: Adams b1. Default: 0.9
        b2: Adams b2. Default: 0.999
        e: Adams epsilon. Default: 1e-6
        weight_decay: Weight decay. Default: 0.01
        max_grad_norm: Maximum norm for the gradients (-1 means no clipping). Default: 1.0
    by xiaodl 
    """

    def __init__(
        self,
        params,
        lr,
        warmup=-1,
        t_total=-1,
        schedule="warmup_linear",
        betas=(0.9, 0.999),
        eps=1e-6,
        weight_decay=0.01,
        max_grad_norm=1.0,
    ):
        if not lr >= 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= warmup < 1.0 and not warmup == -1:
            raise ValueError("Invalid warmup: {} - should be in [0.0, 1.0[ or -1".format(warmup))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(
            lr=lr,
            schedule=schedule,
            warmup=warmup,
            t_total=t_total,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            max_grad_norm=max_grad_norm,
        )
        super(Adamax, self).__init__(params, defaults)

    def get_lr(self):
        lr = []
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if len(state) == 0:
                    return [0]
                if group["t_total"] != -1:
                    schedule_fct = schedule_func(group["schedule"])
                    lr_scheduled = group["lr"] * schedule_fct(
                        state["step"] / group["t_total"], group["warmup"]
                    )
                else:
                    lr_scheduled = group["lr"]
                lr.append(lr_scheduled)
        return lr

    def to(self, device):
        """ Move the optimizer state to a specified device"""
        for state in self.state.values():
            state["exp_avg"].to(device)
            state["exp_inf"].to(device)

    def initialize_step(self, initial_step):
        """Initialize state with a defined step (but we don't have stored averaged).
        Arguments:
            initial_step (int): Initial step number.
        """
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                # State initialization
                state["step"] = initial_step
                # Exponential moving average of gradient values
                state["exp_avg"] = torch.zeros_like(p.data)
                # Exponential moving average of squared gradient values
                state["exp_inf"] = torch.zeros_like(p.data)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        "Adam does not support sparse gradients, please consider SparseAdam instead"
                    )

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p.data)
                    state["exp_inf"] = torch.zeros_like(p.data)

                exp_avg, exp_inf = state["exp_avg"], state["exp_inf"]
                beta1, beta2 = group["betas"]
                eps = group["eps"]
                # Add grad clipping
                if group["max_grad_norm"] > 0:
                    clip_grad_norm_(p, group["max_grad_norm"])

                # Update biased first moment estimate.
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                # Update the exponentially weighted infinity norm.
                norm_buf = torch.cat(
                    [exp_inf.mul_(beta2).unsqueeze(0), grad.abs().add_(eps).unsqueeze_(0)], 0
                )
                torch.max(norm_buf, 0, keepdim=False, out=(exp_inf, exp_inf.new().long()))
                update = exp_avg / (exp_inf + eps)

                if group["weight_decay"] > 0.0:
                    update += group["weight_decay"] * p.data

                if group["t_total"] != -1:
                    schedule_fct = schedule_func(group["schedule"])
                    lr_scheduled = group["lr"] * schedule_fct(
                        state["step"] / group["t_total"], group["warmup"]
                    )
                else:
                    lr_scheduled = group["lr"]

                update_with_lr = lr_scheduled * update
                p.data.add_(-update_with_lr)
                state["step"] += 1

        return loss


class RAdam(Optimizer):
    """Modified from: https://github.com/LiyuanLucasLiu/RAdam/blob/master/radam.py
    """

    def __init__(
        self,
        params,
        lr,
        warmup=-1,
        t_total=-1,
        schedule="warmup_linear",
        betas=(0.9, 0.999),
        eps=1e-6,
        weight_decay=0.001,
        max_grad_norm=1.0,
    ):
        if not lr >= 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= warmup < 1.0 and not warmup == -1:
            raise ValueError("Invalid warmup: {} - should be in [0.0, 1.0[ or -1".format(warmup))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(
            lr=lr,
            schedule=schedule,
            warmup=warmup,
            t_total=t_total,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            max_grad_norm=max_grad_norm,
        )
        self.buffer = [[None, None, None] for ind in range(10)]
        super(RAdam, self).__init__(params, defaults)

    def get_lr(self):
        lr = []
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if len(state) == 0:
                    return [0]
                if group["t_total"] != -1:
                    schedule_fct = schedule_func(group["schedule"])
                    lr_scheduled = group["lr"] * schedule_fct(
                        state["step"] / group["t_total"], group["warmup"]
                    )
                else:
                    lr_scheduled = group["lr"]
                lr.append(lr_scheduled)
        return lr

    def to(self, device):
        """ Move the optimizer state to a specified device"""
        for state in self.state.values():
            state["exp_avg"].to(device)
            state["exp_avg_sq"].to(device)

    def initialize_step(self, initial_step):
        """Initialize state with a defined step (but we don't have stored averaged).
        Arguments:
            initial_step (int): Initial step number.
        """
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                # State initialization
                state["step"] = initial_step
                # Exponential moving average of gradient values
                state["exp_avg"] = torch.zeros_like(p.data)
                # Exponential moving average of squared gradient values
                state["exp_avg_sq"] = torch.zeros_like(p.data)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        # set_trace()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError("RAdam does not support sparse gradients")

                p_data_fp32 = p.data.float()

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p_data_fp32)
                    state["exp_avg_sq"] = torch.zeros_like(p_data_fp32)
                else:
                    state["exp_avg"] = state["exp_avg"].type_as(p_data_fp32)
                    state["exp_avg_sq"] = state["exp_avg_sq"].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]
                eps = group["eps"]
                # Add grad clipping
                if group["max_grad_norm"] > 0:
                    clip_grad_norm_(p, group["max_grad_norm"])

                # Update biased first moment estimate.
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                state["step"] += 1

                if group["t_total"] != -1:
                    schedule_fct = schedule_func(group["schedule"])
                    lr_scheduled = group["lr"] * schedule_fct(
                        state["step"] / group["t_total"], group["warmup"]
                    )
                else:
                    lr_scheduled = group["lr"]

                buffered = self.buffer[int(state["step"] % 10)]
                if state["step"] == buffered[0]:
                    N_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state["step"]
                    beta2_t = beta2 ** state["step"]
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * state["step"] * beta2_t / (1 - beta2_t)
                    buffered[1] = N_sma

                    # more conservative since it's an approximated value
                    if N_sma >= 5:
                        step_size = (
                            lr_scheduled
                            * math.sqrt(
                                (1 - beta2_t)
                                * (N_sma - 4)
                                / (N_sma_max - 4)
                                * (N_sma - 2)
                                / N_sma
                                * N_sma_max
                                / (N_sma_max - 2)
                            )
                            / (1 - beta1 ** state["step"])
                        )
                    else:
                        step_size = lr_scheduled / (1 - beta1 ** state["step"])
                    buffered[2] = step_size

                if N_sma >= 5:
                    denom = exp_avg_sq.sqrt().add_(group["eps"])
                    p_data_fp32.addcdiv_(-step_size, exp_avg, denom)
                else:
                    p_data_fp32.add_(-step_size, exp_avg)

                if group["weight_decay"] != 0:
                    p_data_fp32.add_(-group["weight_decay"] * lr_scheduled, p_data_fp32)

                p.data.copy_(p_data_fp32)

        return loss
