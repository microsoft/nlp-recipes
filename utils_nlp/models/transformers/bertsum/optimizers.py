# Modifications Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# This script reuses code from https://github.com/nlpyang/Presumm


""" Optimizers class """
import torch
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_


# from onmt.utils import use_gpu
# from models.adam import Adam


def use_gpu(opt):
    """
    Creates a boolean if gpu used
    """
    return (hasattr(opt, "gpu_ranks") and len(opt.gpu_ranks) > 0) or (
        hasattr(opt, "gpu") and opt.gpu > -1
    )


def build_optim(model, opt, checkpoint):
    """ Build optimizer """
    saved_optimizer_state_dict = None

    if opt.train_from:
        optim = checkpoint["optim"]
        # We need to save a copy of optim.optimizer.state_dict() for setting
        # the, optimizer state later on in Stage 2 in this method, since
        # the method optim.set_parameters(model.parameters()) will overwrite
        # optim.optimizer, and with ith the values stored in
        # optim.optimizer.state_dict()
        # saved_optimizer_state_dict = optim.optimizer.state_dict()
        saved_optimizer_state_dict = optim
    else:
        optim = Optimizer(
            opt.optim,
            opt.learning_rate,
            opt.max_grad_norm,
            lr_decay=opt.learning_rate_decay,
            start_decay_steps=opt.start_decay_steps,
            decay_steps=opt.decay_steps,
            beta1=opt.adam_beta1,
            beta2=opt.adam_beta2,
            adagrad_accum=opt.adagrad_accumulator_init,
            decay_method=opt.decay_method,
            warmup_steps=opt.warmup_steps,
        )

    optim.set_parameters(model.named_parameters())

    if opt.train_from:
        optim.optimizer.load_state_dict(saved_optimizer_state_dict)
        if use_gpu(opt):
            for state in optim.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

        if (optim.method == "adam") and (len(optim.optimizer.state) < 1):
            raise RuntimeError(
                "Error: loaded Adam optimizer from existing model"
                + " but optimizer state is empty"
            )

    return optim


class MultipleOptimizer(object):
    """ Implement multiple optimizers needed for sparse adam """

    def __init__(self, op):
        """ ? """
        self.optimizers = op

    def zero_grad(self):
        """ ? """
        for op in self.optimizers:
            op.zero_grad()

    def step(self):
        """ ? """
        for op in self.optimizers:
            op.step()

    @property
    def state(self):
        """ ? """
        return {k: v for op in self.optimizers for k, v in op.state.items()}

    def state_dict(self):
        """ ? """
        return [op.state_dict() for op in self.optimizers]

    def load_state_dict(self, state_dicts):
        """ ? """
        assert len(state_dicts) == len(self.optimizers)
        for i in range(len(state_dicts)):
            self.optimizers[i].load_state_dict(state_dicts[i])


class Optimizer(object):
    """
    Controller class for optimization. Mostly a thin
    wrapper for `optim`, but also useful for implementing
    rate scheduling beyond what is currently available.
    Also implements necessary methods for training RNNs such
    as grad manipulations.

    Args:
      method (:obj:`str`): one of [sgd, adagrad, adadelta, adam]
      lr (float): learning rate
      lr_decay (float, optional): learning rate decay multiplier
      start_decay_steps (int, optional): step to start learning rate decay
      beta1, beta2 (float, optional): parameters for adam
      adagrad_accum (float, optional): initialization parameter for adagrad
      decay_method (str, option): custom decay options
      warmup_steps (int, option): parameter for `noam` decay
      model_size (int, option): parameter for `noam` decay

    We use the default parameters for Adam that are suggested by
    the original paper https://arxiv.org/pdf/1412.6980.pdf
    These values are also used by other established implementations,
    e.g. https://www.tensorflow.org/api_docs/python/tf/train/AdamOptimizer
    https://keras.io/optimizers/
    Recently there are slightly different values used in the paper
    "Attention is all you need"
    https://arxiv.org/pdf/1706.03762.pdf, particularly the value beta2=0.98
    was used there however, beta2=0.999 is still arguably the more
    established value, so we use that here as well
    """

    def __init__(
        self,
        method,
        learning_rate,
        max_grad_norm,
        lr_decay=1,
        start_decay_steps=None,
        decay_steps=None,
        beta1=0.9,
        beta2=0.999,
        adagrad_accum=0.0,
        decay_method=None,
        warmup_steps=4000,
        weight_decay=0,
    ):
        self.last_ppl = None
        self.learning_rate = learning_rate
        self.original_lr = learning_rate
        self.max_grad_norm = max_grad_norm
        self.method = method
        self.lr_decay = lr_decay
        self.start_decay_steps = start_decay_steps
        self.decay_steps = decay_steps
        self.start_decay = False
        self._step = 0
        self.betas = [beta1, beta2]
        self.adagrad_accum = adagrad_accum
        self.decay_method = decay_method
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay

    def set_parameters(self, params):
        """ ? """
        self.params = []
        self.sparse_params = []
        for k, p in params:
            if p.requires_grad:
                if self.method != "sparseadam" or "embed" not in k:
                    self.params.append(p)
                else:
                    self.sparse_params.append(p)
        if self.method == "sgd":
            self.optimizer = optim.SGD(self.params, lr=self.learning_rate)
        elif self.method == "adagrad":
            self.optimizer = optim.Adagrad(self.params, lr=self.learning_rate)
            for group in self.optimizer.param_groups:
                for p in group["params"]:
                    self.optimizer.state[p]["sum"] = self.optimizer.state[p][
                        "sum"
                    ].fill_(self.adagrad_accum)
        elif self.method == "adadelta":
            self.optimizer = optim.Adadelta(self.params, lr=self.learning_rate)
        elif self.method == "adam":
            self.optimizer = optim.Adam(
                self.params, lr=self.learning_rate, betas=self.betas, eps=1e-9
            )
        else:
            raise RuntimeError("Invalid optim method: " + self.method)

        self.param_groups = self.optimizer.param_groups
        self.state = self.optimizer.state

    def _set_rate(self, learning_rate):
        self.learning_rate = learning_rate
        if self.method != "sparseadam":
            self.optimizer.param_groups[0]["lr"] = self.learning_rate
        else:
            for op in self.optimizer.optimizers:
                op.param_groups[0]["lr"] = self.learning_rate

    def step(self):
        """Update the model parameters based on current gradients.

        Optionally, will employ gradient modification or update learning
        rate.
        """
        self._step += 1

        # Decay method used in tensor2tensor.
        if self.decay_method == "noam":
            self._set_rate(
                self.original_lr
                * min(self._step ** (-0.5), self._step * self.warmup_steps ** (-1.5))
            )

        else:
            if (self.start_decay_steps is not None) and (
                self._step >= self.start_decay_steps
            ):
                self.start_decay = True
            if self.start_decay:
                if (self._step - self.start_decay_steps) % self.decay_steps == 0:
                    self.learning_rate = self.learning_rate * self.lr_decay

        if self.method != "sparseadam":
            self.optimizer.param_groups[0]["lr"] = self.learning_rate

        if self.max_grad_norm:
            clip_grad_norm_(self.params, self.max_grad_norm)
        self.optimizer.step()

    def add_param_group(self, param_group):
        r"""Add a param group to the :class:`Optimizer` s `param_groups`.

            This can be useful when fine tuning a pre-trained network as frozen layers can be made
            trainable and added to the :class:`Optimizer` as training progresses.

            Arguments:
                param_group (dict): Specifies what Tensors should be optimized along with group
                specific optimization options.
            """
        assert isinstance(param_group, dict), "param group must be a dict"

        params = param_group["params"]
        if isinstance(params, torch.Tensor):
            param_group["params"] = [params]
        elif isinstance(params, set):
            raise TypeError(
                "optimizer parameters need to be organized in ordered collections, but "
                "the ordering of tensors in sets will change between runs. Please use a list instead."
            )
        else:
            param_group["params"] = list(params)

        for param in param_group["params"]:
            if not isinstance(param, torch.Tensor):
                raise TypeError(
                    "optimizer can only optimize Tensors, "
                    "but one of the params is " + torch.typename(param)
                )
            if not param.is_leaf:
                raise ValueError("can't optimize a non-leaf Tensor")

        for name, default in self.defaults.items():
            if default is required and name not in param_group:
                raise ValueError(
                    "parameter group didn't specify a value of required optimization parameter "
                    + name
                )
            else:
                param_group.setdefault(name, default)

        param_set = set()
        for group in self.param_groups:
            param_set.update(set(group["params"]))

        if not param_set.isdisjoint(set(param_group["params"])):
            raise ValueError("some parameters appear in more than one parameter group")

        self.param_groups.append(param_group)

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)

    def state_dict(self):
        """ ? """
        return self.optimizer.state_dict()

    def zero_grad(self):
        """ ? """
        self.optimizer.zero_grad()
