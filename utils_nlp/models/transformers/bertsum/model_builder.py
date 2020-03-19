# Modifications Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# This script reuses code from https://github.com/nlpyang/Presumm

"""
The BertSum models for both extractive and abstractive summarization.
"""

import sys
import copy

import torch
import torch.nn as nn
from transformers import BertModel, BertConfig
from torch.nn.init import xavier_uniform_

from .decoder import TransformerDecoder
from .encoder import Classifier, ExtTransformerEncoder, RNNEncoder
from .optimizers import Optimizer
from .loss import abs_loss


def load_optimizer_checkpoint(optimizer, checkpoint):
    if checkpoint is not None:
        saved_optimizer_state_dict = checkpoint  # .state_dict()
        optimizer.optimizer.load_state_dict(saved_optimizer_state_dict)
        if (optimizer.method == "adam") and (len(optimizer.optimizer.state) < 1):
            raise RuntimeError(
                "Error: loaded Adam optimizer from existing model"
                + " but optimizer state is empty"
            )


def build_optim(
    model,
    optim="adam",
    lr=0.002,
    max_grad_norm=0,
    beta1=0.9,
    beta2=0.999,
    decay_method="noam",
    warmup_steps=8000,
):
    """ Build optimizer """
    optim = Optimizer(
        optim,
        lr,
        max_grad_norm,
        beta1=beta1,
        beta2=beta2,
        decay_method=decay_method,
        warmup_steps=warmup_steps,
    )

    optim.set_parameters(list(model.named_parameters()))

    return optim


def build_optim_bert(
    model,
    optim="adam",
    lr_bert=0.002,
    max_grad_norm=0,
    beta1=0.9,
    beta2=0.999,
    warmup_steps_bert=8000,
):

    optim = Optimizer(
        optim,
        lr_bert,
        max_grad_norm,
        beta1=beta1,
        beta2=beta2,
        decay_method="noam",
        warmup_steps=warmup_steps_bert,
    )

    params = [
        (n, p)
        for n, p in list(model.named_parameters())
        if (n.startswith("bert.model") or n.startswith("module.bert.model"))
    ]
    optim.set_parameters(params)

    return optim


def build_optim_dec(
    model,
    optim="adam",
    lr_dec=0.2,
    max_grad_norm=0,
    beta1=0.9,
    beta2=0.999,
    warmup_steps_dec=8000,
):
    optim = Optimizer(
        optim,
        lr_dec,
        max_grad_norm,
        beta1=beta1,
        beta2=beta2,
        decay_method="noam",
        warmup_steps=warmup_steps_dec,
    )

    params = [
        (n, p)
        for n, p in list(model.named_parameters())
        if (not n.startswith("bert.model") and not n.startswith("module.bert.model"))
    ]
    optim.set_parameters(params)

    return optim


def get_generator(vocab_size, dec_hidden_size):
    gen_func = nn.LogSoftmax(dim=-1)
    generator = nn.Sequential(nn.Linear(dec_hidden_size, vocab_size), gen_func)
    # generator.to(device)

    return generator

class Transformer(nn.Module):
    def __init__(self, temp_dir, model_class, pretrained_model_name, pretrained_config):
        super(Transformer, self).__init__()
        if(pretrained_model_name):
            self.model = model_class.from_pretrained(pretrained_model_name,
                                                   cache_dir=temp_dir)
            #self.model = BertModel.from_pretrained('bert-base-uncased', cache_dir=temp_dir)
        else:
            self.model = model_class(pretrained_config)

    def forward(self, x, segs, mask):
        if "DistilBertModel" in str(type(self.model)):
            outputs = self.model(x, attention_mask =mask)
        else:
            outputs = self.model(x, token_type_ids=segs, attention_mask =mask)
        #print(outputs)
        #print(len(outputs))
        top_vec = outputs[0] 
        
        return top_vec

class BertSumExt(nn.Module):
    def __init__(self, encoder, args, model_class, pretrained_model_name, max_pos=512, pretrained_config = None, temp_dir="./"):
        super(BertSumExt, self).__init__()
        self.loss = torch.nn.BCELoss(reduction='none')
        #self.device = device
        self.transformer = Transformer(temp_dir, model_class, pretrained_model_name, pretrained_config)
        if (encoder == 'classifier'):
            self.encoder = Classifier(self.transformer.model.config.hidden_size)
        elif(encoder=='transformer'):
            self.encoder = ExtTransformerEncoder(self.transformer.model.config.hidden_size, args.ff_size, args.heads,
                                                   args.dropout, args.inter_layers)
        elif(encoder=='rnn'):
            self.encoder = RNNEncoder(bidirectional=True, num_layers=1,
                                      input_size=self.transformer.model.config.hidden_size, hidden_size=args.rnn_size,
                                      dropout=args.dropout)
        elif (encoder == 'baseline'):
            bert_config = BertConfig(self.transformer.model.config.vocab_size, hidden_size=args.hidden_size,
                                     num_hidden_layers=6, num_attention_heads=8, intermediate_size=args.ff_size)
            self.transformer.model = BertModel(bert_config)
            self.encoder = Classifier(self.transformer.model.config.hidden_size)
        
        self.max_pos = max_pos
        if(max_pos > 512):
            my_pos_embeddings = nn.Embedding(self.max_pos, self.transformer.model.config.hidden_size)
            my_pos_embeddings.weight.data[:512] = self.transformer.model.embeddings.position_embeddings.weight.data
            my_pos_embeddings.weight.data[512:] = self.transformer.model.embeddings.position_embeddings.weight.data[-1][None,:].repeat(self.max_pos-512,1)
            self.transformer.model.embeddings.position_embeddings = my_pos_embeddings

        if args.param_init != 0.0:
            for p in self.encoder.parameters():
                p.data.uniform_(-args.param_init, args.param_init)
        if args.param_init_glorot:
            for p in self.encoder.parameters():
                if p.dim() > 1:
                    xavier_uniform_(p)

        #self.to(device)
    def load_cp(self, pt):
        self.load_state_dict(pt['model'], strict=True)

    def forward(self, x, segs, clss, mask, mask_cls, labels=None, sentence_range=None):

        top_vec = self.transformer(x, segs, mask)
        sents_vec = top_vec[torch.arange(top_vec.size(0)).unsqueeze(1), clss]
        sents_vec = sents_vec * mask_cls[:, :, None].float()
        sent_scores = self.encoder(sents_vec, mask_cls).squeeze(-1)
        if labels is not None:
            loss = self.loss(sent_scores, labels.float())
            loss = (loss*mask_cls.float()).sum()
            sent_scores = sent_scores + mask_cls.float()
            return loss, sent_scores, mask_cls
        else:
            sent_scores = sent_scores + mask_cls.float()
            return sent_scores, mask_cls




class Bert(nn.Module):
    def __init__(self, large, temp_dir, finetune=False):
        super(Bert, self).__init__()
        if large:
            self.model = BertModel.from_pretrained(
                "bert-large-uncased", cache_dir=temp_dir
            )
        else:
            self.model = BertModel.from_pretrained(
                "bert-base-uncased", cache_dir=temp_dir
            )

        self.finetune = finetune

    def forward(self, x, segs, mask):
        if self.finetune:
            outputs = self.model(x, attention_mask=mask)
        else:
            self.eval()
            with torch.no_grad():
                outputs = self.model(x, attention_mask=mask)
        top_vec = outputs[0]
        return top_vec


class AbsSummarizer(nn.Module):
    def __init__(
        self,
        large=False,
        symbols=None,
        temp_dir="./",
        finetune_bert=True,
        encoder="bert",
        max_pos=512,
        use_bert_emb=True,
        share_emb=False,
        dec_dropout=0.2,
        dec_layers=6,
        dec_hidden_size=768,
        dec_heads=8,
        dec_ff_size=2048,
        enc_hidden_size=512,
        enc_ff_size=512,
        enc_dropout=0.2,
        enc_layers=6,
        label_smoothing=0.1,
        checkpoint=None,
        bert_from_extractive=None,
        test=False,
    ):
        super(AbsSummarizer, self).__init__()
        self.bert = Bert(large, temp_dir, finetune_bert)

        if bert_from_extractive is not None:
            self.bert.model.load_state_dict(
                dict(
                    [
                        (n[11:], p)
                        for n, p in bert_from_extractive.items()
                        if n.startswith("bert.model")
                    ]
                ),
                strict=True,
            )

        if encoder == "baseline":
            bert_config = BertConfig(
                self.bert.model.config.vocab_size,
                hidden_size=enc_hidden_size,
                num_hidden_layers=enc_layers,
                num_attention_heads=8,
                intermediate_size=enc_ff_size,
                hidden_dropout_prob=enc_dropout,
                attention_probs_dropout_prob=enc_dropout,
            )
            self.bert.model = BertModel(bert_config)

        if max_pos > 512:
            my_pos_embeddings = nn.Embedding(
                max_pos, self.bert.model.config.hidden_size
            )
            my_pos_embeddings.weight.data[
                :512
            ] = self.bert.model.embeddings.position_embeddings.weight.data
            my_pos_embeddings.weight.data[
                512:
            ] = self.bert.model.embeddings.position_embeddings.weight.data[-1][
                None, :
            ].repeat(
                max_pos - 512, 1
            )
            self.bert.model.embeddings.position_embeddings = my_pos_embeddings
        self.vocab_size = self.bert.model.config.vocab_size
        tgt_embeddings = nn.Embedding(
            self.vocab_size, self.bert.model.config.hidden_size, padding_idx=0
        )
        if share_emb:
            tgt_embeddings.weight = copy.deepcopy(
                self.bert.model.embeddings.word_embeddings.weight
            )

        self.decoder = TransformerDecoder(
            dec_layers,
            dec_hidden_size,
            heads=dec_heads,
            d_ff=dec_ff_size,
            dropout=dec_dropout,
            embeddings=tgt_embeddings,
        )

        self.generator = get_generator(self.vocab_size, dec_hidden_size)
        self.generator[0].weight = self.decoder.embeddings.weight

        for module in self.decoder.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                module.weight.data.normal_(mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        for p in self.generator.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)
            else:
                p.data.zero_()
        if use_bert_emb:
            tgt_embeddings = nn.Embedding(
                self.vocab_size, self.bert.model.config.hidden_size, padding_idx=0
            )
            tgt_embeddings.weight = copy.deepcopy(
                self.bert.model.embeddings.word_embeddings.weight
            )
            self.decoder.embeddings = tgt_embeddings
            self.generator[0].weight = self.decoder.embeddings.weight

        self.symbols = symbols
        self.label_smoothing = label_smoothing
        self.test = test
        if not test:
            self.train_loss = abs_loss(
                self.generator,
                self.symbols,
                self.vocab_size,
                train=True,
                label_smoothing=self.label_smoothing,
            )

    def load_checkpoint(self, checkpoint):
        if checkpoint is not None:
            self.load_state_dict(checkpoint, strict=False)
        if not self.test:
            self.train_loss = abs_loss(
                self.generator,
                self.symbols,
                self.vocab_size,
                train=True,
                label_smoothing=self.label_smoothing,
            )

    # def move_to_device(self, device, move_to_device_fn):
    # self.to(device)
    # self.generator = move_to_device_fn(self.generator, device)
    #    self = move_to_device_fn(self, device)
    #    return self

    # def forward(self, src, tgt, segs, clss, mask_src, mask_tgt, mask_cls):
    def forward(
        self, src, segs, mask_src, tgt, tgt_num_tokens
    ):  # , mask_tgt, mask_cls):
        top_vec = self.bert(src, segs, mask_src)
        dec_state = self.decoder.init_decoder_state(src, top_vec)
        decoder_outputs, state = self.decoder(tgt[:, :-1], top_vec, dec_state)
        loss = self.train_loss.monolithic_compute_loss(
            decoder_outputs, tgt[:, 1:], tgt_num_tokens
        )
        return loss, decoder_outputs
