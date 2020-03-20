"""
Portions Copyright (c) Microsoft Corporation
"""
import gc
import itertools
import random
import torch


class IterableDistributedSampler(object):
    """ Distributed sampler for iterable dataset.

    Args:
        world_size (int, optional): Total number of GPUs that will be used.
            Defaults to 1.
        rank (int, optional): Rank of the current GPU. Defaults to 0.
        local_rank(int, optional): local_rank of the current GPU. Defaults to -1.

    """

    def __init__(self, world_size=1, rank=0, local_rank=-1):
        self.world_size = world_size
        self.rank = rank
        self.local_rank = local_rank

    def iter(self, iterable):
        if self.local_rank == -1:
            return iterable
        if self.world_size > 1:
            return itertools.islice(iterable, self.rank, None, self.world_size)
        else:
            return iterable


class ChunkDataLoader(object):
    """ Data Loader for Chunked Dataset.

    Args:
        datasets (list): list of data item list.
        batch_size (int): Number of tokens per batch.
        shuffle (bool): Whether the data is shuffled.
        is_labeled (bool): Whether the data is labeled.
        sampler (obj): Data sampler.

    """

    def __init__(self, datasets, batch_size, shuffle, is_labeled, sampler):
        self.datasets = datasets
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.is_labeled = is_labeled
        self.cur_iter = self._next_dataset_iterator(datasets)
        assert self.cur_iter is not None
        self.sampler = sampler

    def eachiter(self):
        dataset_iter = (d for d in self.datasets)
        while self.cur_iter is not None:
            for batch in self.cur_iter:
                yield batch
            self.cur_iter = self._next_dataset_iterator(dataset_iter)

    def __iter__(self):
        return self.sampler.iter(self.eachiter())

    def _next_dataset_iterator(self, dataset_iter):
        try:
            # Drop the current dataset for decreasing memory
            if hasattr(self, "cur_dataset"):
                self.cur_dataset = None
                gc.collect()
                del self.cur_dataset
                gc.collect()

            self.cur_dataset = next(dataset_iter)
        except StopIteration:
            return None

        return DataIterator(
            dataset=self.cur_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            is_labeled=self.is_labeled,
        )


class Batch(object):
    def _pad(self, data, pad_id, width=-1):
        if width == -1:
            width = max(len(d) for d in data)
        rtn_data = [d + [pad_id] * (width - len(d)) for d in data]
        return rtn_data

    def __init__(self, data=None, is_labeled=False):
        """Create a Batch from a list of examples."""
        if data is None or len(data) == 0:
            raise ValueError("data is empty")
        self.batch_size = len(data)
        pre_src = [x[0] for x in data]
        pre_segs = [x[2] for x in data]
        pre_clss = [x[3] for x in data]

        src = torch.tensor(self._pad(pre_src, 0))

        pre_labels = None
        labels = None
        if is_labeled:
            pre_labels = [x[1] for x in data]
            labels = torch.tensor(self._pad(pre_labels, 0))
        segs = torch.tensor(self._pad(pre_segs, 0))
        mask = ~(src == 0)

        clss = torch.tensor(self._pad(pre_clss, -1))
        mask_cls = ~(clss == -1)
        clss[clss == -1] = 0

        setattr(self, "clss", clss)
        setattr(self, "mask_cls", mask_cls)
        setattr(self, "src", src)
        setattr(self, "segs", segs)
        setattr(self, "mask", mask)

        src_str = [x[4] for x in data]
        setattr(self, "src_str", src_str)

        if is_labeled:
            setattr(self, "labels", labels)
            tgt_str = [x[5] for x in data]
            setattr(self, "tgt_str", tgt_str)

    def to(self, device):
        src = self.src.to(device)
        segs = self.segs.to(device)
        clss = self.clss.to(device)
        mask = self.mask.to(device)
        mask_cls = self.mask_cls.to(device)

        setattr(self, "clss", clss)
        setattr(self, "mask_cls", mask_cls)
        setattr(self, "src", src)
        setattr(self, "segs", segs)
        setattr(self, "mask", mask)
        if hasattr(self, "labels"):
            labels = self.labels.to(device)
            setattr(self, "labels", labels.to(device))

        return self

    def __len__(self):
        return self.batch_size


def create_batch_with_size(data, batch_size):
    """Yield elements from data in chunks of batch_size."""
    minibatch, size_so_far = [], 0
    for ex in data:
        minibatch.append(ex)
        size_so_far = simple_batch_size_fn(ex, len(minibatch))
        if size_so_far == batch_size:
            yield minibatch
            minibatch, size_so_far = [], 0
        elif size_so_far > batch_size:
            yield minibatch[:-1]
            minibatch, size_so_far = minibatch[-1:], simple_batch_size_fn(ex, 1)
    if minibatch:
        yield minibatch


def simple_batch_size_fn(new, count):
    src, labels = new[0], new[1]
    global max_n_sents, max_n_tokens, max_size
    if count == 1:
        max_size = 0
        max_n_sents = 0
        max_n_tokens = 0
    max_n_sents = max(max_n_sents, len(src))
    max_size = max(max_size, max_n_sents)
    src_elements = count * max_size
    return src_elements


class DataIterator(object):
    def __init__(self, dataset, batch_size, is_labeled=False, shuffle=True, sort=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.is_labeled = is_labeled
        self.iterations = 0
        self.shuffle = shuffle
        self.sort = sort

        self.sort_key = lambda x: len(x[1])

        self._iterations_this_epoch = 0

    def data(self):
        if self.shuffle:
            random.shuffle(self.dataset)
        xs = self.dataset
        return xs

    def preprocess(self, ex, is_labeled):
        src = ex["src"]
        if "labels" in ex:
            labels = ex["labels"]
        else:
            labels = None  # ex['src_sent_labels']

        segs = ex["segs"]
        # if(not self.args.use_interval):
        #    segs=[0]*len(segs)
        clss = ex["clss"]
        src_txt = ex["src_txt"]
        tgt_txt = ex["tgt_txt"]

        if is_labeled:
            return src, labels, segs, clss, src_txt, tgt_txt
        else:
            return src, labels, segs, clss, src_txt, None

    def batch_buffer(self, data, batch_size):
        minibatch, size_so_far = [], 0
        for ex in data:
            if len(ex["src"]) == 0:
                continue
            ex = self.preprocess(ex, self.is_labeled)
            if ex is None:
                continue
            minibatch.append(ex)
            size_so_far = simple_batch_size_fn(ex, len(minibatch))
            if size_so_far == batch_size:
                yield minibatch
                minibatch, size_so_far = [], 0
            elif size_so_far > batch_size:
                yield minibatch[:-1]
                minibatch, size_so_far = minibatch[-1:], simple_batch_size_fn(ex, 1)
        if minibatch:
            yield minibatch

    def create_batches(self):
        """ Create batches """
        data = self.data()
        for buffer in self.batch_buffer(data, self.batch_size * 50):

            if self.sort:
                p_batch = sorted(buffer, key=lambda x: len(x[3]))
            else:
                p_batch = buffer
            p_batch = create_batch_with_size(p_batch, self.batch_size)

            p_batch = list(p_batch)
            if self.shuffle:
                random.shuffle(p_batch)
            for b in p_batch:
                yield b

    def __iter__(self):
        while True:
            self.batches = self.create_batches()
            for idx, minibatch in enumerate(self.batches):
                # fast-forward if loaded from state
                if self._iterations_this_epoch > idx:
                    continue
                self.iterations += 1
                self._iterations_this_epoch += 1
                # if len(minibatch) == 0:
                #    continue
                batch = Batch(minibatch, self.is_labeled)
                yield batch
            return
