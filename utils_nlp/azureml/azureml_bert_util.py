# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# Original source:
# https://github.com/microsoft/AzureML-BERT/blob/dec79be13befdd51fa72c05419cf9288d32eb263/finetune/PyTorch/azureml_bert_util.py

"""
    Classes and helper functions for fine-tuning BERT models at scale (e.g.
    distributed training) using AzureML.
"""


from horovod.torch.mpi_ops import allreduce_async_, synchronize
import horovod.torch as hvd
import torch

from collections import OrderedDict

try:
    from apex_C import flatten
    from apex_C import unflatten
except ImportError:
    try:
        _ = warned_flatten
    except NameError:
        print(
            "Warning:  apex was installed without --cpp_ext.  Falling back to Python flatten and "
            "unflatten."
        )
        warned_flatten = True
    from torch._utils import _flatten_dense_tensors as flatten
    from torch._utils import _unflatten_dense_tensors as unflatten


def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x / warmup
    return 1.0 - x


def adjust_gradient_accumulation_steps(x, initial_steps, target_steps, warmup):
    return min(max(int(x / warmup * target_steps), initial_steps), target_steps)


class DistributedCommunicator:
    """ Assists in making communication with multiple nodes for distributed training"""

    def __init__(self, accumulation_step=1):
        hvd.init()
        self.local_rank = hvd.local_rank()
        self.world_size = hvd.size()
        self.rank = hvd.rank()
        self.n_gpu = torch.cuda.device_count()
        self.node_count = self.world_size // self.n_gpu
        self.accumulation_step = accumulation_step
        self.count_down = accumulation_step - 1
        self._multi_node = self.node_count > 1
        if not self._multi_node:
            # use PyTorch build-in NCCL backend for single node training
            torch.distributed.init_process_group(
                backend="nccl",
                init_method="tcp://127.0.0.1:6000",
                world_size=self.n_gpu,
                rank=self.local_rank,
            )

    def register_model(self, model, fp16):
        #  broadcast model parameters
        if self.node_count > 1:
            hvd.broadcast_parameters(model.state_dict(), root_rank=0)
        else:
            for param in model.parameters():
                torch.distributed.broadcast_multigpu([param], 0)

        # register hook for reduce when backpropagate
        self._parameter_names = {v: k for k, v in sorted(model.named_parameters())}
        self._handles = {}
        self._requires_update = set()
        self._grad_accs = []
        self._grad = []
        self._compression = hvd.Compression.fp16 if fp16 else hvd.Compression.none
        for p in model.parameters():
            if p.requires_grad:
                p.grad = p.data.new(p.size()).zero_()
                self._requires_update.add(p)
                p_tmp = p.expand_as(p)
                grad_acc = p_tmp.grad_fn.next_functions[0][0]
                grad_acc.register_hook(self._make_hook(p))
                self._grad_accs.append(grad_acc)

    def _allreduce_tensor(self, p):
        assert p not in self._handles
        assert not p.grad.requires_grad
        tensor = p.grad
        name = self._parameter_names.get(p)
        if self._multi_node:
            tensor_compressed, ctx = self._compression.compress(tensor)
            handle = allreduce_async_(tensor_compressed, average=True, name=name)
            self._handles[p] = (handle, ctx)
        else:
            self._handles[p] = tensor

    def _make_hook(self, p):
        def hook(*ignore):
            if self.count_down == 0:
                self._allreduce_tensor(p)

        return hook

    def synchronize(self):
        synced = False
        if self.count_down == 0:
            missing_p = self._requires_update - set(self._handles.keys())
            for p in missing_p:
                self._allreduce_tensor(p)

            if self._multi_node:
                for p, value in self._handles.items():
                    handle, ctx = value
                    output = synchronize(handle)
                    p.grad.set_(self._compression.decompress(output, ctx) / self.accumulation_step)
            else:
                buckets = OrderedDict()
                for tensor in self._handles.values():
                    tp = tensor.type()
                    if tp not in buckets:
                        buckets[tp] = []
                    buckets[tp].append(tensor)
                for tp in buckets:
                    bucket = buckets[tp]
                    coalesced = flatten(bucket) / self.world_size / self.accumulation_step
                    torch.distributed.all_reduce_multigpu([coalesced])
                    for buf, synced in zip(bucket, unflatten(coalesced, bucket)):
                        buf.copy_(synced)
            self._handles.clear()
            synced = True
            self.count_down = self.accumulation_step

        self.count_down -= 1
        return synced

    def set_accumulation_step(self, accumulation_step):
        self.accumulation_step = accumulation_step
        self.count_down = self.accumulation_step - 1
