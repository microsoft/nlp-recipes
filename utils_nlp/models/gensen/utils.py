#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Minibatching utilities."""
import itertools
import operator
import os
import pickle

import numpy as np
import torch
from sklearn.utils import shuffle
from torch.autograd import Variable

# Change to python3+.
# from itertools import zip


class DataIterator(object):
    """Data Iterator."""

    @staticmethod
    def _trim_vocab(vocab, vocab_size):
        """Discard start, end, pad and unk tokens if already present.

        Args:
            vocab(list): Vocabulary.
            vocab_size(int): The size of the vocabulary.

        Returns:
            word2id(list): Word to index list.
            id2word(list): Index to word list.
        """
        if "<s>" in vocab:
            del vocab["<s>"]
        if "<pad>" in vocab:
            del vocab["<pad>"]
        if "</s>" in vocab:
            del vocab["</s>"]
        if "<unk>" in vocab:
            del vocab["<unk>"]

        word2id = {"<s>": 0, "<pad>": 1, "</s>": 2, "<unk>": 3}

        id2word = {0: "<s>", 1: "<pad>", 2: "</s>", 3: "<unk>"}

        sorted_word2id = sorted(
            vocab.items(), key=operator.itemgetter(1), reverse=True
        )

        if vocab_size != -1:
            sorted_words = [x[0] for x in sorted_word2id[:vocab_size]]
        else:
            sorted_words = [x[0] for x in sorted_word2id]

        for ind, word in enumerate(sorted_words):
            word2id[word] = ind + 4

        for ind, word in enumerate(sorted_words):
            id2word[ind + 4] = word

        return word2id, id2word

    def construct_vocab(
        self, sentences, vocab_size, lowercase=False, charlevel=False
    ):
        """Create vocabulary.

        Args:
            sentences(list): The list of sentences.
            vocab_size(int): The size of vocabulary.
            lowercase(bool): If lowercase the sentences.
            charlevel(bool): If need to split the sentence with space.

        Returns:
            word2id(list): Word to index list.
            id2word(list): Index to word list.
        """
        vocab = {}
        for sentence in sentences:
            if isinstance(sentence, str):
                if lowercase:
                    sentence = sentence.lower()
                if not charlevel:
                    sentence = sentence.split()
            for word in sentence:
                if word not in vocab:
                    vocab[word] = 1
                else:
                    vocab[word] += 1
        word2id, id2word = self._trim_vocab(vocab, vocab_size)
        return word2id, id2word


class BufferedDataIterator(DataIterator):
    """Multi Parallel corpus data iterator."""

    def __init__(
        self,
        src,
        trg,
        src_vocab_size,
        trg_vocab_size,
        tasknames,
        save_dir,
        buffer_size=1e6,
        lowercase=False,
        seed=0,
    ):
        """Initialize params.

        Args:
            src(list): source dataset.
            trg(list): target dataset.
            src_vocab_size(int): The size of source vocab.
            trg_vocab_size(int): The size of target vocab.
            tasknames(list): The list of task names.
            save_dir(str): The saving dir.
            buffer_size(float): Buffer size.
            lowercase(bool): if lowercase the data.
        """
        self.seed = seed
        self.fname_src = src
        self.fname_trg = trg
        self.src_vocab_size = src_vocab_size
        self.trg_vocab_size = trg_vocab_size
        self.tasknames = tasknames
        self.save_dir = save_dir
        self.buffer_size = buffer_size
        self.lowercase = lowercase

        # Open a list of file pointers to all the files.
        self.f_src = [
            open(fname, "r", encoding="utf-8") for fname in self.fname_src
        ]
        self.f_trg = [
            open(fname, "r", encoding="utf-8") for fname in self.fname_trg
        ]

        # Initialize dictionaries that contain sentences & word mapping dicts
        self.src = [
            {"data": [], "word2id": None, "id2word": None}
            for i in range(len(self.fname_src))
        ]
        self.trg = [
            {"data": [], "word2id": None, "id2word": None}
            for i in range(len(self.fname_trg))
        ]
        self.build_vocab()

        """Reset file pointers to the start after reading the file to
        build vocabularies."""
        for idx in range(len(self.src)):
            self._reset_filepointer(idx)
        for idx in range(len(self.src)):
            self.fetch_buffer(idx)

    def _reset_filepointer(self, idx):
        """Reset file pointer.

        Args:
            idx(int): Index used to reset file pointer.

        """
        self.f_src[idx] = open(self.fname_src[idx], "r", encoding="utf-8")
        self.f_trg[idx] = open(self.fname_trg[idx], "r", encoding="utf-8")

    def fetch_buffer(self, idx, reset=True):
        """Fetch sentences from the file into the buffer.

        Args:
            idx(int): Index used to fetch the sentences.
            reset(bool): If need to reset the contents of the current buffer.

        """
        # Reset the contents of the current buffer.
        if reset:
            self.src[idx]["data"] = []
            self.trg[idx]["data"] = []

        # Populate buffer
        for src, trg in zip(self.f_src[idx], self.f_trg[idx]):
            if len(self.src[idx]["data"]) == self.buffer_size:
                break
            if self.lowercase:
                self.src[idx]["data"].append(src.lower().split())
                self.trg[idx]["data"].append(trg.lower().split())

            else:
                self.src[idx]["data"].append(src.split())
                self.trg[idx]["data"].append(trg.split())

        # Sort sentences by decreasing length (hacky bucketing)
        self.src[idx]["data"], self.trg[idx]["data"] = zip(
            *sorted(
                zip(self.src[idx]["data"], self.trg[idx]["data"]),
                key=lambda x: len(x[0]),
                reverse=True,
            )
        )

        """If buffer isn't full after reading the contents of the file,
        cycle around. """
        if len(self.src[idx]["data"]) < self.buffer_size:
            assert len(self.src[idx]["data"]) == len(self.trg[idx]["data"])
            # Cast things to list to avoid issue with calling .append above
            self.src[idx]["data"] = list(self.src[idx]["data"])
            self.trg[idx]["data"] = list(self.trg[idx]["data"])
            self._reset_filepointer(idx)
            self.fetch_buffer(idx, reset=False)

    def build_vocab(self):
        """Build a memory efficient vocab."""
        # Construct common source vocab.
        # Check if save directory exists.
        if not os.path.exists(self.save_dir):
            raise ValueError("Could not find save dir : %s" % self.save_dir)

        # Check if a cached vocab file exists.
        if os.path.exists(os.path.join(self.save_dir, "src_vocab.pkl")):
            vocab = pickle.load(
                open(os.path.join(self.save_dir, "src_vocab.pkl"), "rb")
            )
            word2id, id2word = vocab["word2id"], vocab["id2word"]
        # If not, compute the vocab from scratch and store a cache.
        else:
            word2id, id2word = self.construct_vocab(
                itertools.chain.from_iterable(self.f_src),
                self.src_vocab_size,
                self.lowercase,
            )
            pickle.dump(
                {"word2id": word2id, "id2word": id2word},
                open(os.path.join(self.save_dir, "src_vocab.pkl"), "wb"),
            )
        for corpus in self.src:
            corpus["word2id"], corpus["id2word"] = word2id, id2word

        # Do the same for the target vocabulary.
        if os.path.exists(os.path.join(self.save_dir, "trg_vocab.pkl")):
            vocab = pickle.load(
                open(os.path.join(self.save_dir, "trg_vocab.pkl"), "rb")
            )
            for idx, (corpus, fname) in enumerate(zip(self.trg, self.f_trg)):
                word2id, id2word = (
                    vocab[self.tasknames[idx]]["word2id"],
                    vocab[self.tasknames[idx]]["id2word"],
                )
                corpus["word2id"], corpus["id2word"] = word2id, id2word
        else:
            trg_vocab_dump = {}
            for idx, (corpus, fname) in enumerate(zip(self.trg, self.f_trg)):
                word2id, id2word = self.construct_vocab(
                    fname, self.trg_vocab_size, self.lowercase
                )
                corpus["word2id"], corpus["id2word"] = word2id, id2word
                trg_vocab_dump[self.tasknames[idx]] = {}
                trg_vocab_dump[self.tasknames[idx]]["word2id"] = word2id
                trg_vocab_dump[self.tasknames[idx]]["id2word"] = id2word

            pickle.dump(
                trg_vocab_dump,
                open(os.path.join(self.save_dir, "trg_vocab.pkl"), "wb"),
            )

    def shuffle_dataset(self, idx):
        """Shuffle current buffer."""
        self.src[idx]["data"], self.trg[idx]["data"] = shuffle(
            self.src[idx]["data"],
            self.trg[idx]["data"],
            random_state=self.seed,
        )

    def get_parallel_minibatch(
        self, corpus_idx, index, batch_size, max_len_src, max_len_trg
    ):
        """Prepare minibatch.

        Args:
            corpus_idx(int): Corpus Index.
            index(int): Index.
            batch_size(int): Batch Size.
            max_len_src(int): Max length for resource.
            max_len_trg(int): Max length ofr target.

        Returns: minibatch of src-trg pairs(dict).

        """
        src_lines = [
            ["<s>"] + line[: max_len_src - 2] + ["</s>"]
            for line in self.src[corpus_idx]["data"][
                index : index + batch_size
            ]
        ]

        trg_lines = [
            ["<s>"] + line[: max_len_trg - 2] + ["</s>"]
            for line in self.trg[corpus_idx]["data"][
                index : index + batch_size
            ]
        ]

        """Sort sentences by decreasing length within a minibatch for
        `torch.nn.utils.packed_padded_sequence`"""
        src_lens = [len(line) for line in src_lines]
        sorted_indices = np.argsort(src_lens)[::-1]

        sorted_src_lines = [src_lines[idx] for idx in sorted_indices]
        sorted_trg_lines = [trg_lines[idx] for idx in sorted_indices]

        sorted_src_lens = [len(line) for line in sorted_src_lines]
        sorted_trg_lens = [len(line) for line in sorted_trg_lines]

        max_src_len = max(sorted_src_lens)
        max_trg_len = max(sorted_trg_lens)

        # Map words to indices
        input_lines_src = [
            [
                self.src[corpus_idx]["word2id"][w]
                if w in self.src[corpus_idx]["word2id"]
                else self.src[corpus_idx]["word2id"]["<unk>"]
                for w in line
            ]
            + [self.src[corpus_idx]["word2id"]["<pad>"]]
            * (max_src_len - len(line))
            for line in sorted_src_lines
        ]

        input_lines_trg = [
            [
                self.trg[corpus_idx]["word2id"][w]
                if w in self.trg[corpus_idx]["word2id"]
                else self.trg[corpus_idx]["word2id"]["<unk>"]
                for w in line[:-1]
            ]
            + [self.trg[corpus_idx]["word2id"]["<pad>"]]
            * (max_trg_len - len(line))
            for line in sorted_trg_lines
        ]

        output_lines_trg = [
            [
                self.trg[corpus_idx]["word2id"][w]
                if w in self.trg[corpus_idx]["word2id"]
                else self.trg[corpus_idx]["word2id"]["<unk>"]
                for w in line[1:]
            ]
            + [self.trg[corpus_idx]["word2id"]["<pad>"]]
            * (max_trg_len - len(line))
            for line in sorted_trg_lines
        ]

        # Cast lists to torch tensors
        input_lines_src = Variable(torch.LongTensor(input_lines_src)).cuda()
        input_lines_trg = Variable(torch.LongTensor(input_lines_trg)).cuda()
        output_lines_trg = Variable(torch.LongTensor(output_lines_trg)).cuda()
        sorted_src_lens = (
            Variable(torch.LongTensor(sorted_src_lens), volatile=True)
            .squeeze()
            .cuda()
        )

        # Return minibatch of src-trg pairs
        return {
            "input_src": input_lines_src,
            "input_trg": input_lines_trg,
            "output_trg": output_lines_trg,
            "src_lens": sorted_src_lens,
            "type": "seq2seq",
        }


class NLIIterator(DataIterator):
    """Data iterator for tokenized NLI datasets."""

    def __init__(
        self, train, dev, test, vocab_size, lowercase=True, vocab=None, seed=0
    ):
        """Initialize params.

        Each of train/dev/test is a tab-separate file of the form
        premise \t hypothesis \t label.

        Args:
            train(torch.Tensor): Training dataset.
            dev(torch.Tensor): Validation dataset.
            test(torch.Tensor): Testing dataset.
            vocab_size(int): The size of the vocabulary.
            lowercase(bool): If lowercase the dataset.
            vocab(Union[bytes,str): The list of the vocabulary.
        """
        self.seed = seed
        self.train = train
        self.dev = dev
        self.test = test
        self.vocab_size = vocab_size
        self.lowercase = lowercase
        self.vocab = vocab
        self.train_lines = [
            line.strip().lower().split("\t")
            for line in open(self.train, encoding="utf-8")
        ]
        self.dev_lines = [
            line.strip().lower().split("\t")
            for line in open(self.dev, encoding="utf-8")
        ]
        self.test_lines = [
            line.strip().lower().split("\t")
            for line in open(self.test, encoding="utf-8")
        ]

        if self.vocab is not None:
            # binary mode doesn't take an encoding argument
            self.vocab = pickle.load(open(self.vocab, "rb"))
            self.word2id = self.vocab["word2id"]
            self.id2word = self.vocab["id2word"]
            self.vocab_size = len(self.word2id)
        else:
            self.word2id, self.id2word = self.construct_vocab(
                [x[0] for x in self.train_lines]
                + [x[1] for x in self.train_lines],
                self.vocab_size,
                lowercase=self.lowercase,
            )

        # Label text to class mapping.
        self.text2label = {"entailment": 0, "neutral": 1, "contradiction": 2}

        self.shuffle_dataset()

    def shuffle_dataset(self):
        """Shuffle training data."""
        self.train_lines = shuffle(self.train_lines, random_state=self.seed)

    def get_parallel_minibatch(self, index, batch_size, sent_type="train"):
        """Prepare minibatch.

        Args:
            index(int): The index for line.
            batch_size(int): Batch size.
            sent_type(str): Type of dataset.

        Returns:
            dict for batch training.
        """
        if sent_type == "train":
            lines = self.train_lines
        elif sent_type == "dev":
            lines = self.dev_lines
        else:
            lines = self.test_lines

        sent1 = [
            ["<s>"] + line[0].split() + ["</s>"]
            for line in lines[index : index + batch_size]
        ]

        sent2 = [
            ["<s>"] + line[1].split() + ["</s>"]
            for line in lines[index : index + batch_size]
        ]

        labels = [
            self.text2label[line[2]]
            for line in lines[index : index + batch_size]
        ]

        sent1_lens = [len(line) for line in sent1]
        sorted_sent1_indices = np.argsort(sent1_lens)[::-1]
        sorted_sent1_lines = [sent1[idx] for idx in sorted_sent1_indices]
        rev_sent1 = np.argsort(sorted_sent1_indices)

        sent2_lens = [len(line) for line in sent2]
        sorted_sent2_indices = np.argsort(sent2_lens)[::-1]
        sorted_sent2_lines = [sent2[idx] for idx in sorted_sent2_indices]
        rev_sent2 = np.argsort(sorted_sent2_indices)

        sorted_sent1_lens = [len(line) for line in sorted_sent1_lines]
        sorted_sent2_lens = [len(line) for line in sorted_sent2_lines]

        max_sent1_len = max(sorted_sent1_lens)
        max_sent2_len = max(sorted_sent2_lens)

        sent1 = [
            [
                self.word2id[w] if w in self.word2id else self.word2id["<unk>"]
                for w in line
            ]
            + [self.word2id["<pad>"]] * (max_sent1_len - len(line))
            for line in sorted_sent1_lines
        ]

        sent2 = [
            [
                self.word2id[w] if w in self.word2id else self.word2id["<unk>"]
                for w in line
            ]
            + [self.word2id["<pad>"]] * (max_sent2_len - len(line))
            for line in sorted_sent2_lines
        ]

        sent1 = Variable(torch.LongTensor(sent1)).cuda()
        sent2 = Variable(torch.LongTensor(sent2)).cuda()
        labels = Variable(torch.LongTensor(labels)).cuda()
        sent1_lens = (
            Variable(torch.LongTensor(sorted_sent1_lens), requires_grad=False)
            .squeeze()
            .cuda()
        )
        sent2_lens = (
            Variable(torch.LongTensor(sorted_sent2_lens), requires_grad=False)
            .squeeze()
            .cuda()
        )
        rev_sent1 = (
            Variable(torch.LongTensor(rev_sent1), requires_grad=False)
            .squeeze()
            .cuda()
        )
        rev_sent2 = (
            Variable(torch.LongTensor(rev_sent2), requires_grad=False)
            .squeeze()
            .cuda()
        )

        return {
            "sent1": sent1,
            "sent2": sent2,
            "sent1_lens": sent1_lens,
            "sent2_lens": sent2_lens,
            "rev_sent1": rev_sent1,
            "rev_sent2": rev_sent2,
            "labels": labels,
            "type": "nli",
        }


def get_validation_minibatch(
    src, trg, index, batch_size, src_word2id, trg_word2id
):
    """Prepare minibatch.

    Args:
        src(list): source data.
        trg(list): target data.
        index(int): index for the file.
        batch_size(int): batch size.
        src_word2id(list): Word to index for source.
        trg_word2id(list): Word to index for target.

    Returns:
        Dict for seq2seq model.
    """
    src_lines = [
        ["<s>"] + line + ["</s>"] for line in src[index : index + batch_size]
    ]

    trg_lines = [
        ["<s>"] + line + ["</s>"] for line in trg[index : index + batch_size]
    ]

    src_lens = [len(line) for line in src_lines]
    sorted_indices = np.argsort(src_lens)[::-1]

    sorted_src_lines = [src_lines[idx] for idx in sorted_indices]
    sorted_trg_lines = [trg_lines[idx] for idx in sorted_indices]

    sorted_src_lens = [len(line) for line in sorted_src_lines]
    sorted_trg_lens = [len(line) for line in sorted_trg_lines]

    max_src_len = max(sorted_src_lens)
    max_trg_len = max(sorted_trg_lens)

    input_lines_src = [
        [src_word2id[w] if w in src else src_word2id["<unk>"] for w in line]
        + [src_word2id["<pad>"]] * (max_src_len - len(line))
        for line in sorted_src_lines
    ]

    input_lines_trg = [
        [
            trg_word2id[w] if w in trg_word2id else trg_word2id["<unk>"]
            for w in line[:-1]
        ]
        + [trg_word2id["<pad>"]] * (max_trg_len - len(line))
        for line in sorted_trg_lines
    ]

    output_lines_trg = [
        [
            trg_word2id[w] if w in trg_word2id else trg_word2id["<unk>"]
            for w in line[1:]
        ]
        + [trg_word2id["<pad>"]] * (max_trg_len - len(line))
        for line in sorted_trg_lines
    ]
    # For pytroch 0.4
    with torch.no_grad():
        input_lines_src = Variable(torch.LongTensor(input_lines_src)).cuda()
        input_lines_trg = Variable(torch.LongTensor(input_lines_trg)).cuda()
        output_lines_trg = Variable(torch.LongTensor(output_lines_trg)).cuda()
        # sorted_src_lens = Variable(
        #     torch.LongTensor(sorted_src_lens)
        # ).squeeze().cuda()
        sorted_src_lens = (
            Variable(torch.LongTensor(sorted_src_lens))
            .view(len(sorted_src_lens))
            .cuda()
        )
    return {
        "input_src": input_lines_src,
        "input_trg": input_lines_trg,
        "output_trg": output_lines_trg,
        "src_lens": sorted_src_lens,
        "type": "seq2seq",
    }


def compute_validation_loss(
    config, model, train_iterator, criterion, task_idx, lowercase=False
):
    """Compute validation loss for a task.

    Args:
        config(dict): configuration list.
        model(MultitaskModel): model.
        train_iterator(BufferedDataIterator): Multi Parallel corpus data iterator.
        criterion(nn.CrossEntropyLoss): criterion function for loss.
        task_idx(int): Task index.
        lowercase(bool): If lowercase the data.

    Returns: float as the mean of the loss.

    """
    val_src = config["data"]["paths"][task_idx]["val_src"]
    val_trg = config["data"]["paths"][task_idx]["val_trg"]

    if lowercase:
        val_src = [
            line.strip().lower().split()
            for line in open(val_src, "r", encoding="utf-8")
        ]
        val_trg = [
            line.strip().lower().split()
            for line in open(val_trg, "r", encoding="utf-8")
        ]
    else:
        val_src = [
            line.strip().split()
            for line in open(val_src, "r", encoding="utf-8")
        ]
        val_trg = [
            line.strip().split()
            for line in open(val_trg, "r", encoding="utf-8")
        ]

    batch_size = config["training"]["batch_size"]
    losses = []
    for j in range(0, len(val_src), batch_size):
        minibatch = get_validation_minibatch(
            val_src,
            val_trg,
            j,
            batch_size,
            train_iterator.src[task_idx]["word2id"],
            train_iterator.trg[task_idx]["word2id"],
        )
        decoder_logit = model(minibatch, task_idx)

        loss = criterion(
            decoder_logit.contiguous().view(-1, decoder_logit.size(2)),
            minibatch["output_trg"].contiguous().view(-1),
        )

        # losses.append(loss.data[0])
        losses.append(loss.item())

    return np.mean(losses)


# Original source: https://github.com/Maluuba/gensen
