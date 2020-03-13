# Modifications Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# This script reuses code from https://github.com/nlpyang/Presumm

""" Translator Class and builder """
from __future__ import print_function
import codecs
import os
import math

import torch
from torch import nn
from tensorboardX import SummaryWriter

# from others.utils import rouge_results_to_str, test_rouge, tile
from .beam import GNMTGlobalScorer


def build_predictor(
    tokenizer,
    symbols,
    model,
    alpha=0.6,
    beam_size=5,
    min_length=15,
    max_length=150,
    logger=None,
):
    scorer = GNMTGlobalScorer(alpha, length_penalty="wu")

    translator = Translator(
        beam_size,
        min_length,
        max_length,
        model,
        tokenizer,
        symbols,
        global_scorer=scorer,
        logger=logger,
    )
    return translator


def tile(x, count, dim=0):
    """
    Tiles x on dimension dim count times.
    """
    perm = list(range(len(x.size())))
    if dim != 0:
        perm[0], perm[dim] = perm[dim], perm[0]
        x = x.permute(perm).contiguous()
    out_size = list(x.size())
    out_size[0] *= count
    batch = x.size(0)
    x = (
        x.view(batch, -1)
        .transpose(0, 1)
        .repeat(count, 1)
        .transpose(0, 1)
        .contiguous()
        .view(*out_size)
    )
    if dim != 0:
        x = x.permute(perm).contiguous()
    return x


class Translator(nn.Module):
    """
    Uses a model to translate a batch of sentences.


    Args:
       model (:obj:`onmt.modules.NMTModel`):
          NMT model to use for translation
       fields (dict of Fields): data fields
       beam_size (int): size of beam to use
       n_best (int): number of translations produced
       max_length (int): maximum length output to produce
       global_scores (:obj:`GlobalScorer`):
         object to rescore final translations
       copy_attn (bool): use copy attention during translation
       beam_trace (bool): trace beam search for debugging
       logger(logging.Logger): logger.
    """

    def __init__(
        self,
        beam_size,
        min_length,
        max_length,
        model,
        vocab,
        symbols,
        block_trigram=True,
        global_scorer=None,
        logger=None,
        dump_beam="",
    ):
        super(Translator, self).__init__()
        self.logger = logger

        self.model = model.module if hasattr(model, "module") else model
        self.generator = self.model.generator
        self.decoder = self.model.decoder
        self.bert = self.model.bert

        self.vocab = vocab
        self.symbols = symbols
        self.start_token = symbols["BOS"]
        self.end_token = symbols["EOS"]

        self.global_scorer = global_scorer
        self.beam_size = beam_size
        self.min_length = min_length
        self.max_length = max_length
        self.block_trigram = block_trigram

        self.dump_beam = dump_beam

        # for debugging
        self.beam_trace = self.dump_beam != ""
        self.beam_accum = None

        if self.beam_trace:
            self.beam_accum = {
                "predicted_ids": [],
                "beam_parent_ids": [],
                "scores": [],
                "log_probs": [],
            }

    """
    def eval(self):
        self.model.eval()
        self.bert.eval()
        self.decoder.eval()
        self.generator.eval()
    """

    def forward(self, src, segs, mask_src):
        """
        Translate a batch of sentences.

        Mostly a wrapper around :obj:`Beam`.

        Args:
           batch (:obj:`Batch`): a batch from a dataset object
           data (:obj:`Dataset`): the dataset object
           fast (bool): enables fast beam search (may not support all features)

        Todo:
           Shouldn't need the original dataset.
        """
        with torch.no_grad():
            predictions, scores = self._fast_translate_batch(
                src, segs, mask_src, self.max_length, min_length=self.min_length
            )
            return predictions, scores

    def _fast_translate_batch(self, src, segs, mask_src, max_length, min_length=0):
        # TODO: faster code path for beam_size == 1.

        # TODO: support these blacklisted features.
        assert not self.dump_beam

        beam_size = self.beam_size
        batch_size = src.size()[0]  # 32 #batch.batch_size

        src_features = self.bert(src, segs, mask_src)
        this_decoder = (
            self.decoder.module if hasattr(self.decoder, "module") else self.decoder
        )
        dec_states = this_decoder.init_decoder_state(src, src_features, with_cache=True)

        device = src_features.device

        # Tile states and memory beam_size times.
        dec_states.map_batch_fn(lambda state, dim: tile(state, beam_size, dim=dim))
        src_features = tile(src_features, beam_size, dim=0)
        batch_offset = torch.arange(batch_size, dtype=torch.long, device=device)
        beam_offset = torch.arange(
            0, batch_size * beam_size, step=beam_size, dtype=torch.long, device=device
        )
        alive_seq = torch.full(
            [batch_size * beam_size, 1],
            self.start_token,
            dtype=torch.long,
            device=device,
        )

        # Give full probability to the first beam on the first step.
        topk_log_probs = torch.tensor(
            [0.0] + [float("-inf")] * (beam_size - 1), device=device
        ).repeat(batch_size)

        # Structure that holds finished hypotheses.
        hypotheses = [[] for _ in range(batch_size)]  # noqa: F812

        results = {}
        results["predictions"] = [[] for _ in range(batch_size)]  # noqa: F812
        results["scores"] = [[] for _ in range(batch_size)]  # noqa: F812
        # results["gold_score"] = [0] * batch_size
        # results["batch"] = batch

        for step in range(max_length):
            decoder_input = alive_seq[:, -1].view(1, -1)

            # Decoder forward.
            decoder_input = decoder_input.transpose(0, 1)

            dec_out, dec_states = this_decoder(
                decoder_input, src_features, dec_states, step=step
            )

            # Generator forward.
            log_probs = self.generator.forward(dec_out.transpose(0, 1).squeeze(0))
            vocab_size = log_probs.size(-1)

            if step < min_length:
                log_probs[:, self.end_token] = torch.Tensor([-1e20])

            # Multiply probs by the beam probability.
            log_probs += topk_log_probs.view(-1).unsqueeze(1)

            alpha = self.global_scorer.alpha
            length_penalty = ((5.0 + (step + 1)) / 6.0) ** alpha

            # Flatten probs into a list of possibilities.
            curr_scores = log_probs / length_penalty

            if self.block_trigram:
                cur_len = alive_seq.size(1)
                if cur_len > 3:
                    for i in range(alive_seq.size(0)):
                        fail = False
                        words = [int(w) for w in alive_seq[i]]
                        words = [self.vocab.ids_to_tokens[w] for w in words]
                        words = " ".join(words).replace(" ##", "").split()
                        if len(words) <= 3:
                            continue
                        trigrams = [
                            (words[i - 1], words[i], words[i + 1])
                            for i in range(1, len(words) - 1)
                        ]
                        trigram = tuple(trigrams[-1])
                        if trigram in trigrams[:-1]:
                            fail = True
                        if fail:
                            curr_scores[i] = torch.Tensor([-10e20])

            curr_scores = curr_scores.reshape(-1, beam_size * vocab_size)
            topk_scores, topk_ids = curr_scores.topk(beam_size, dim=-1)

            # Recover log probs.
            topk_log_probs = topk_scores * length_penalty

            # Resolve beam origin and true word ids.
            topk_beam_index = topk_ids.div(vocab_size)
            topk_ids = topk_ids.fmod(vocab_size)

            # Map beam_index to batch_index in the flat representation.
            batch_index = topk_beam_index + beam_offset[
                : topk_beam_index.size(0)
            ].unsqueeze(1)
            select_indices = batch_index.view(-1)

            # Append last prediction.
            alive_seq = torch.cat(
                [alive_seq.index_select(0, select_indices), topk_ids.view(-1, 1)], -1
            )

            is_finished = topk_ids.eq(self.end_token)
            if step + 1 == max_length:
                is_finished.fill_(True)
            # End condition is top beam is finished.
            end_condition = is_finished[:, 0].eq(True)
            if step + 1 == max_length:
                assert not any(end_condition.eq(False))

            # Save finished hypotheses.
            if is_finished.any():
                predictions = alive_seq.view(-1, beam_size, alive_seq.size(-1))
                for i in range(is_finished.size(0)):
                    b = batch_offset[i]
                    if end_condition[i]:
                        is_finished[i].fill_(1)
                    finished_hyp = is_finished[i].nonzero().view(-1)
                    # Store finished hypotheses for this batch.
                    for j in finished_hyp:
                        hypotheses[b].append((topk_scores[i, j], predictions[i, j, 1:]))
                    # If the batch reached the end, save the n_best hypotheses.
                    if end_condition[i]:
                        best_hyp = sorted(
                            hypotheses[b], key=lambda x: x[0], reverse=True
                        )
                        score, pred = best_hyp[0]
                        results["scores"][b].append(score)
                        results["predictions"][b].append(pred)
                non_finished = end_condition.eq(0).nonzero().view(-1)
                # If all sentences are translated, no need to go further.
                if len(non_finished) == 0:
                    break
                # Remove finished batches for the next step.
                topk_log_probs = topk_log_probs.index_select(0, non_finished)
                batch_index = batch_index.index_select(0, non_finished)
                batch_offset = batch_offset.index_select(0, non_finished)
                alive_seq = predictions.index_select(0, non_finished).view(
                    -1, alive_seq.size(-1)
                )
            # Reorder states.
            select_indices = batch_index.view(-1)
            src_features = src_features.index_select(0, select_indices)
            dec_states.map_batch_fn(
                lambda state, dim: state.index_select(dim, select_indices)
            )

        empty_output = [len(results["predictions"][b]) <= 0 for b in batch_offset]
        predictions = torch.tensor(
            [
                i[0].tolist()[0 : self.max_length]
                + [0] * (self.max_length - i[0].size()[0])
                for i in results["predictions"]
            ],
            device=device,
        )
        scores = torch.tensor([i[0].item() for i in results["scores"]], device=device)
        return predictions, scores
