# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import h5py
from sklearn.linear_model import LinearRegression
import nltk
import numpy as np
import pickle
import os
import copy
import logging
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


class Encoder(nn.Module):
    """GenSen Encoder.
    
    Original source in https://github.com/Maluuba/gensen
    """

    def __init__(
        self, vocab_size, embedding_dim, hidden_dim, num_layers, rnn_type="GRU"
    ):
        """Initialize all the parameters.

        Args:
            vocab_size (int): Size of the vocabulary.
            embedding_dim (int): the size of each embedding vector
            hidden_dim (int): the size of each hidden vector
            num_layers (int): the number of layers.
            rnn_type (str): Type of RNN.
        """
        super(Encoder, self).__init__()
        self.rnn_type = rnn_type
        rnn = getattr(nn, rnn_type)
        self.src_embedding = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=embedding_dim
        )

        self.encoder = rnn(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )

    def set_pretrained_embeddings(self, embedding_matrix):
        """Set embedding weights.

        Args:
            embedding_matrix(torch.Tensor): Embedding matrix.

        """
        if embedding_matrix.shape[0] != self.src_embedding.weight.size(
            0
        ) or embedding_matrix.shape[1] != self.src_embedding.weight.size(1):
            logging.info(
                """
                Warning pretrained embedding shape mismatch %d x %d
                expected %d x %d"""
                % (
                    embedding_matrix.shape[0],
                    embedding_matrix.shape[1],
                    self.src_embedding.weight.size(0),
                    self.src_embedding.weight.size(1),
                )
            )
            self.src_embedding = nn.Embedding(
                embedding_matrix.shape[0], embedding_matrix.shape[1]
            )
            self.src_vocab_size = embedding_matrix.shape[0]
            self.src_emb_dim = embedding_matrix.shape[1]

        try:
            self.src_embedding.weight.data.set_(
                torch.from_numpy(embedding_matrix)
            )
        except BaseException:
            self.src_embedding.weight.data.set_(
                torch.from_numpy(embedding_matrix).cuda()
            )

        self.src_embedding.cuda()

    def forward(self, input, lengths, return_all=False, pool="last"):
        """Propogate input through the encoder.

        Args:
            input(torch.Tensor): Embedding matrix
            lengths (torch.Tensor): list of sequences lengths of each batch element.
            return_all (bool): Return all or only the last hidden state.
            pool(str): Type of getting hidden state.

        Returns:
            torch.Tensor: Return last or all hidden states.

        """
        embedding = self.src_embedding(input)
        src_emb = pack_padded_sequence(embedding, lengths, batch_first=True)
        if self.rnn_type == "LSTM":
            h, (h_t, _) = self.encoder(src_emb)
        else:
            h, h_t = self.encoder(src_emb)

        # Get hidden state via max-pooling or h_t
        if pool == "last":
            h_t = torch.cat((h_t[-1], h_t[-2]), 1)
        elif pool == "max":
            h_tmp, _ = pad_packed_sequence(h, batch_first=True)
            h_t = torch.max(h_tmp, 1)[0].squeeze()
        else:
            raise ValueError("Pool %s is not valid " % (pool))

        # Return all or only the last hidden state
        if return_all:
            h, _ = pad_packed_sequence(h, batch_first=True)
            return h, h_t
        else:
            return h_t


class GenSen(nn.Module):
    """A wrapper class for multiple GenSen models.
    
    Original source in https://github.com/Maluuba/gensen
    """

    def __init__(self, *args, **kwargs):
        super(GenSen, self).__init__()
        self.gensen_models = args

    def vocab_expansion(self, task_vocab):
        """Expand the model's vocabulary with pretrained word embeddings.

        Args:
            task_vocab(list): Vocabulary for each task.

        """
        for model in self.gensen_models:
            model.vocab_expansion(task_vocab)

    def get_representation(
        self,
        sentences,
        pool="last",
        tokenize=False,
        return_numpy=True,
        add_start_end=True,
    ):
        """ Get model representations.

        Args:
            sentences(list): Sentences to get embeddings.
            pool(str): Type of getting hidden state.
            tokenize(bool): To tokenize or not.
            return_numpy(bool): To return a numpy array or not.
            add_start_end(bool): To add start and end notation (<s> </s>) to each
            sentence or not.

        Returns:
            torch.Tensor : Return last or all hidden states.

        """
        representations = [
            model.get_representation(
                sentences,
                pool=pool,
                tokenize=tokenize,
                return_numpy=return_numpy,
                add_start_end=add_start_end,
            )
            for model in self.gensen_models
        ]
        if return_numpy:
            return (
                np.concatenate([x[0] for x in representations], axis=2),
                np.concatenate([x[1] for x in representations], axis=1),
            )
        else:
            return (
                torch.cat([x[0] for x in representations], 2),
                torch.cat([x[1] for x in representations], 1),
            )


class GenSenSingle(nn.Module):
    """GenSen Wrapper.
    
    Original source in https://github.com/Maluuba/gensen
    """

    def __init__(
        self,
        model_folder,
        filename_prefix,
        pretrained_emb,
        cuda=False,
        rnn_type="GRU",
    ):
        """ Initialize params.

        Args:
            model_folder(str): Folder where the model resides.
            filename_prefix(str): Prefix for the model file.
            pretrained_emb(torch.Tensor): Pretrained Embedding vector.
            cuda(bool): Use Cuda or not.
            rnn_type(str): Type of RNN.
        """

        super(GenSenSingle, self).__init__()
        self.model_folder = model_folder
        self.filename_prefix = filename_prefix
        self.pretrained_emb = pretrained_emb
        self.cuda = cuda
        self.rnn_type = rnn_type
        self._load_params()
        self.vocab_expanded = False

    def _load_params(self):
        """Load pretrained params."""
        # Read vocab pickle files
        open(
            os.path.join(
                self.model_folder, "%s_vocab.pkl" % self.filename_prefix
            ),
            "rb",
        )
        model_vocab = pickle.load(
            open(
                os.path.join(
                    self.model_folder, "%s_vocab.pkl" % self.filename_prefix
                ),
                "rb",
            ),
            encoding="latin1",
        )

        # Word to index mappings
        self.word2id = model_vocab["word2id"]
        self.id2word = model_vocab["id2word"]
        self.task_word2id = self.word2id
        self.id2word = self.id2word

        encoder_model = torch.load(
            os.path.join(
                self.model_folder, "%s.model" % (self.filename_prefix)
            )
        )

        # Initialize encoders
        self.encoder = Encoder(
            vocab_size=encoder_model["src_embedding.weight"].size(0),
            embedding_dim=encoder_model["src_embedding.weight"].size(1),
            hidden_dim=encoder_model["encoder.weight_hh_l0"].size(1),
            num_layers=1 if len(encoder_model) < 10 else 2,
            rnn_type=self.rnn_type,
        )

        # Load pretrained sentence encoder weights
        self.encoder.load_state_dict(encoder_model)

        # Set encoders in eval model.
        self.encoder.eval()

        # Store the initial word embeddings somewhere to re-train vocab expansion multiple times.
        self.model_embedding_matrix = copy.deepcopy(
            self.encoder.src_embedding.weight.data.cpu().numpy()
        )

        # Move encoder to GPU if self.cuda
        if self.cuda:
            self.encoder = self.encoder.cuda()

    def first_expansion(self):
        """ Training linear regression model for the first time."""

        # Read pre-trained word embedding h5 file
        logging.info("Loading pretrained word embeddings")
        pretrained_embeddings = h5py.File(self.pretrained_emb)
        pretrained_embedding_matrix = pretrained_embeddings["embedding"].value
        pretrain_vocab = pretrained_embeddings["words_flatten"].value.split(
            "\n"
        )
        pretrain_word2id = {
            word: ind for ind, word in enumerate(pretrain_vocab)
        }

        # Set up training data for vocabulary expansion
        model_train = []
        pretrain_train = []

        for word in pretrain_word2id:
            if word in self.word2id:
                model_train.append(
                    self.model_embedding_matrix[self.word2id[word]]
                )
                pretrain_train.append(
                    pretrained_embedding_matrix[pretrain_word2id[word]]
                )

        logging.info("Training vocab expansion on model")
        lreg = LinearRegression()
        lreg.fit(pretrain_train, model_train)
        self.lreg = lreg
        self.pretrain_word2id = pretrain_word2id
        self.pretrained_embedding_matrix = pretrained_embedding_matrix

    def vocab_expansion(self, task_vocab):
        """ Expand the model's vocabulary with pretrained word embeddings.

        Args:
            task_vocab(list): Vocabulary for each task.
        """

        self.task_word2id = {"<s>": 0, "<pad>": 1, "</s>": 2, "<unk>": 3}

        self.task_id2word = {0: "<s>", 1: "<pad>", 2: "</s>", 3: "<unk>"}

        ctr = 4
        for idx, word in enumerate(task_vocab):
            if word not in self.task_word2id:
                self.task_word2id[word] = ctr
                self.task_id2word[ctr] = word
                ctr += 1

        if not self.vocab_expanded:
            self.first_expansion()

        # Expand vocabulary using the linear regression model
        task_embeddings = []
        oov_pretrain = 0
        oov_task = 0

        for word in self.task_id2word.values():
            if word in self.word2id:
                task_embeddings.append(
                    self.model_embedding_matrix[self.word2id[word]]
                )
            elif word in self.pretrain_word2id:
                oov_task += 1
                task_embeddings.append(
                    self.lreg.predict(
                        self.pretrained_embedding_matrix[
                            self.pretrain_word2id[word]
                        ].reshape(1, -1)
                    )
                    .squeeze()
                    .astype(np.float32)
                )
            else:
                oov_pretrain += 1
                oov_task += 1
                task_embeddings.append(
                    self.model_embedding_matrix[self.word2id["<unk>"]]
                )

        logging.info("Found %d task OOVs " % (oov_task))
        logging.info("Found %d pretrain OOVs " % (oov_pretrain))
        task_embeddings = np.stack(task_embeddings)
        self.encoder.set_pretrained_embeddings(task_embeddings)
        self.vocab_expanded = True

        # Move encoder to GPU if self.cuda
        if self.cuda:
            self.encoder = self.encoder.cuda()

    def get_minibatch(self, sentences, tokenize=False, add_start_end=True):
        """Prepare minibatch.

        Args:
            sentences(list): Sentences to get embeddings.
            tokenize(bool): To tokenize or not.
            add_start_end(bool): To add start and end notation (<s> </s>) to each
            sentence or not.

        Returns:
            dict: A dictionary with sentences, lengths and sentence embeddings.

        """
        if tokenize:
            sentences = [
                nltk.word_tokenize(sentence) for sentence in sentences
            ]
        else:
            sentences = [sentence.split() for sentence in sentences]

        if add_start_end:
            sentences = [
                ["<s>"] + sentence + ["</s>"] for sentence in sentences
            ]

        lens = [len(sentence) for sentence in sentences]
        sorted_idx = np.argsort(lens)[::-1]
        sorted_sentences = [sentences[idx] for idx in sorted_idx]
        rev = np.argsort(sorted_idx)
        sorted_lens = [len(sentence) for sentence in sorted_sentences]
        max_len = max(sorted_lens)

        sentences = [
            [
                self.task_word2id[w]
                if w in self.task_word2id
                else self.task_word2id["<unk>"]
                for w in sentence
            ]
            + [self.task_word2id["<pad>"]] * (max_len - len(sentence))
            for sentence in sorted_sentences
        ]

        with torch.no_grad():
            sentences = Variable(torch.LongTensor(sentences))
            rev = Variable(torch.LongTensor(rev))
        lengths = sorted_lens

        if self.cuda:
            sentences = sentences.cuda()
            rev = rev.cuda()

        return {"sentences": sentences, "lengths": lengths, "rev": rev}

    def get_representation(
        self,
        sentences,
        pool="last",
        tokenize=False,
        return_numpy=True,
        add_start_end=True,
    ):
        """Get model representations.

        Args:
            sentences(list): Sentences to get embeddings.
            pool(str): Type of getting hidden state.
            tokenize(bool): To tokenize or not.
            return_numpy(bool): To return a numpy array or not.
            add_start_end(bool): To add start and end notation (<s> </s>) to each
            sentence or not.

        Returns:
            torch.Tensor : Return last or all hidden states.

        """

        minibatch = self.get_minibatch(
            sentences, tokenize=tokenize, add_start_end=add_start_end
        )
        h, h_t = self.encoder(
            input=minibatch["sentences"],
            lengths=minibatch["lengths"],
            return_all=True,
            pool=pool,
        )
        h = h.index_select(0, minibatch["rev"])
        h_t = h_t.index_select(0, minibatch["rev"])
        if return_numpy:
            return h.data.cpu().numpy(), h_t.data.cpu().numpy()
        else:
            return h, h_t


