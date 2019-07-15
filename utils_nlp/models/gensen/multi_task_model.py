# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Parent model for Multitask Training."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from utils_nlp.models.pytorch_modules.conditional_gru import ConditionalGRU


class MultitaskModel(nn.Module):
    """A Multi Task Sequence to Sequence (Seq2Seq) model with GRUs.

    Auxiliary NLI task trained jointly as well.
    Ref: Multi-Task Sequence to Sequence Learning
    https://arxiv.org/pdf/1511.06114.pdf
    """

    def __init__(
        self,
        src_emb_dim,
        trg_emb_dim,
        src_vocab_size,
        trg_vocab_size,
        src_hidden_dim,
        trg_hidden_dim,
        pad_token_src,
        pad_token_trg,
        num_tasks,
        bidirectional=False,
        nlayers_src=1,
        dropout=0.0,
        paired_tasks=None,
    ):
        """Initialize Seq2Seq Model."""
        super(MultitaskModel, self).__init__()
        self.src_vocab_size = src_vocab_size
        self.trg_vocab_size = trg_vocab_size
        self.src_emb_dim = src_emb_dim
        self.trg_emb_dim = trg_emb_dim
        self.src_hidden_dim = src_hidden_dim
        self.trg_hidden_dim = trg_hidden_dim
        self.bidirectional = bidirectional
        self.nlayers_src = nlayers_src
        self.dropout = dropout
        self.num_tasks = num_tasks
        self.paired_tasks = paired_tasks
        self.num_directions = 2 if bidirectional else 1
        self.pad_token_src = pad_token_src
        self.pad_token_trg = pad_token_trg
        self.src_hidden_dim = (
            src_hidden_dim // 2 if self.bidirectional else src_hidden_dim
        )
        self.decoder = ConditionalGRU

        self.src_embedding = nn.Embedding(
            src_vocab_size, src_emb_dim, self.pad_token_src
        )

        self.encoder = nn.GRU(
            src_emb_dim,
            self.src_hidden_dim,
            self.nlayers_src,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=self.dropout,
        )

        self.enc_drp = nn.Dropout(self.dropout)

        self.trg_embedding = nn.ModuleList(
            [
                nn.Embedding(trg_vocab_size, trg_emb_dim, self.pad_token_trg)
                for task in range(self.num_tasks)
            ]
        )

        self.decoders = nn.ModuleList(
            [
                self.decoder(trg_emb_dim, trg_hidden_dim, dropout=self.dropout)
                for task in range(self.num_tasks)
            ]
        )

        self.decoder2vocab = nn.ModuleList(
            [
                nn.Linear(trg_hidden_dim, trg_vocab_size)
                for task in range(self.num_tasks)
            ]
        )

        self.nli_decoder = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(4 * src_hidden_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 3),
        )

        self.init_weights()

    def init_weights(self):
        """Initialize weights."""
        initrange = 0.1
        self.src_embedding.weight.data.uniform_(-initrange, initrange)
        for module in self.trg_embedding:
            module.weight.data.uniform_(-initrange, initrange)
        for module in self.decoder2vocab:
            module.bias.data.fill_(0)

    def set_pretrained_embeddings(self, embedding_matrix):
        """Set embedding weights."""
        if embedding_matrix.shape[0] != self.src_embedding.weight.size(
            0
        ) or embedding_matrix.shape[1] != self.src_embedding.weight.size(1):
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

    def forward(
        self, minibatch, task_idx, return_hidden=False, paired_trg=None
    ):
        """Propogate input through the network.

        Seq2Seq:
        inputs: minibatch['input_src'], minibatch['input_trg']
        input_src       - batch size x source sequence length
        input_trg       - batch size x target sequence length
        src_lengths     - batch size (list)
        paired_trg      - batch size x target sequence length or None
        returns: decoder_logit (pre-softmax over words)
        decoder_logit   - batch size x target sequence length x target vocab size

        NLI:
        sent1           - batch size x source sequence length
        sent2           - batch size x target sequence length
        sent1_lengths   - batch size (list)
        sent2_lengths   - batch size (list)
        rev_sent1       - batch size (LongTensor)
        rev_sent2       - batch size (LongTensor)
        returns: class_logits (pre-softmax over NLI classes)
        decoder_logit   - batch size x 3
        """
        if minibatch["type"] == "nli":
            sent1_emb = self.src_embedding(minibatch["sent1"])
            sent2_emb = self.src_embedding(minibatch["sent2"])

            sent1_lengths = minibatch["sent1_lens"].data.view(-1).tolist()
            sent1_emb = pack_padded_sequence(
                sent1_emb, sent1_lengths, batch_first=True
            )
            sent1, sent1_h = self.encoder(sent1_emb)

            sent2_lengths = minibatch["sent2_lens"].data.view(-1).tolist()
            sent2_emb = pack_padded_sequence(
                sent2_emb, sent2_lengths, batch_first=True
            )
            sent2, sent2_h = self.encoder(sent2_emb)

            if self.bidirectional:
                sent1_h = torch.cat((sent1_h[-1], sent1_h[-2]), 1)
                sent2_h = torch.cat((sent2_h[-1], sent2_h[-2]), 1)
            else:
                sent1_h = sent1_h[-1]
                sent2_h = sent2_h[-1]

            sent1_h = sent1_h.index_select(0, minibatch["rev_sent1"])
            sent2_h = sent2_h.index_select(0, minibatch["rev_sent2"])

            features = torch.cat(
                (
                    sent1_h,
                    sent2_h,
                    torch.abs(sent1_h - sent2_h),
                    sent1_h * sent2_h,
                ),
                1,
            )

            if return_hidden:
                return sent1_h, sent2_h, self.nli_decoder(features)
            else:
                return self.nli_decoder(features)

        else:
            src_emb = self.src_embedding(minibatch["input_src"])
            trg_emb = self.trg_embedding[task_idx](minibatch["input_trg"])
            src_lengths = minibatch["src_lens"].data.view(-1).tolist()
            src_emb = pack_padded_sequence(
                src_emb, src_lengths, batch_first=True
            )

            _, src_h_t = self.encoder(src_emb)

            if self.bidirectional:
                h_t = torch.cat((src_h_t[-1], src_h_t[-2]), 1)
            else:
                h_t = src_h_t[-1]

            h_t = h_t.unsqueeze(0)
            h_t = self.enc_drp(h_t)

            # Debug with squeeze on error.
            trg_h, _ = self.decoders[task_idx](
                trg_emb,
                h_t.view(-1, self.trg_hidden_dim),
                h_t.view(-1, self.trg_hidden_dim),
            )

            trg_h_reshape = trg_h.contiguous().view(
                trg_h.size(0) * trg_h.size(1), trg_h.size(2)
            )

            decoder_logit = self.decoder2vocab[task_idx](trg_h_reshape)
            decoder_logit = decoder_logit.view(
                trg_h.size(0), trg_h.size(1), decoder_logit.size(1)
            )

            if (
                self.paired_tasks is not None
                and task_idx in self.paired_tasks
                and paired_trg is not None
            ):
                other_task_idx = self.paired_tasks[task_idx]
                trg_emb_2 = self.trg_embedding[other_task_idx](paired_trg)

                trg_h_2, _ = self.decoders[other_task_idx](
                    trg_emb_2, h_t.squeeze(), h_t.squeeze()
                )

                trg_h_reshape_2 = trg_h_2.contiguous().view(
                    trg_h_2.size(0) * trg_h_2.size(1), trg_h_2.size(2)
                )

                decoder_logit_2 = self.decoder2vocab[other_task_idx](
                    trg_h_reshape_2
                )
                decoder_logit_2 = decoder_logit_2.view(
                    trg_h_2.size(0), trg_h_2.size(1), decoder_logit_2.size(1)
                )
                if return_hidden:
                    return decoder_logit, decoder_logit_2, h_t
                else:
                    return decoder_logit, decoder_logit_2

            if return_hidden:
                return decoder_logit, h_t
            else:
                return decoder_logit

    def decode(self, logits):
        """Return probability distribution over words."""
        logits_reshape = logits.view(-1, logits.size(2))
        word_probs = F.softmax(logits_reshape)
        word_probs = word_probs.view(
            logits.size(0), logits.size(1), logits.size(2)
        )
        return word_probs

    def get_hidden(self, input_src, src_lengths, strategy="last"):
        """Return the encoder hidden state."""
        src_emb = self.src_embedding(input_src)
        src_lengths = src_lengths.data.view(-1).tolist()
        src_emb = pack_padded_sequence(src_emb, src_lengths, batch_first=True)
        src_h, src_h_t = self.encoder(src_emb)
        if strategy == "last":
            if self.bidirectional:
                h_t = torch.cat((src_h_t[-1], src_h_t[-2]), 1)
            else:
                h_t = src_h_t[-1]
        else:
            src_h, _ = pad_packed_sequence(src_h, batch_first=True)
            h_t = torch.max(src_h, 1)[0].squeeze()

        return src_h, h_t


# Original source: https://github.com/Maluuba/gensen
