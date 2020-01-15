# coding=utf-8
# Copyright (c) Microsoft. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy
from torch.nn.utils import weight_norm
from torch.nn.parameter import Parameter
from .common import activation, init_wrapper
from .dropout_wrapper import DropoutWrapper

class DotProduct(nn.Module):
    def __init__(self, x1_dim, x2_dim, prefix='sim', opt={}, dropout=None):
        super(DotProduct, self).__init__()
        assert x1_dim == x2_dim
        self.opt = opt
        self.prefix = prefix
        self.scale_on = opt.get('{}_scale'.format(self.prefix), False)
        self.scalor = 1.0 / numpy.power(x2_dim, 0.5)

    def forward(self, x1, x2):
        assert x1.size(2) == x2.size(2)
        scores = x1.bmm(x2.transpose(1, 2))
        if self.scale_on:
            scores *= self.scalor
        return scores


class DotProductProject(nn.Module):
    def __init__(self, x1_dim, x2_dim, prefix='sim', opt={}, dropout=None):
        super(DotProductProject, self).__init__()
        self.prefix = prefix
        self.opt = opt
        self.hidden_size = opt.get('{}_hidden_size'.format(self.prefix), 64)
        self.residual_on = opt.get('{}_residual_on'.format(self.prefix), False)
        self.layer_norm_on = opt.get('{}_norm_on'.format(self.prefix), False)
        self.share = opt.get('{}_share'.format(self.prefix), False)
        self.f = activation(opt.get('{}_activation'.format(self.prefix), 'relu'))
        self.scale_on = opt.get('{}_scale_on'.format(self.prefix), False)
        self.dropout = dropout
        x1_in_dim = x1_dim
        x2_in_dim = x2_dim
        out_dim = self.hidden_size
        self.proj_1 = nn.Linear(x1_in_dim, out_dim, bias=False)
        if self.layer_norm_on:
            self.proj_1 = weight_norm(self.proj_1)
        if self.share and x1_in_dim == x2_in_dim:
            self.proj_2 = self.proj_1
        else:
            self.proj_2 = nn.Linear(x2_in_dim, out_dim)
            if self.layer_norm_on:
                self.proj_2 = weight_norm(self.proj_2)

        if self.scale_on:
            self.scalar = Parameter(torch.ones(1,1,1) / (self.hidden_size ** 0.5), requires_grad=False)
        else:
            self.sclalar = Parameter(torch.ones(1,1, self.hidden_size), requires_grad=True)

    def forward(self, x1, x2):
        assert x1.size(2) == x2.size(2)
        if self.dropout:
            x1 = self.dropout(x1)
            x2 = self.dropout(x2)
        x1_flat = x1.contiguous().view(-1, x1.size(2))
        x2_flat = x2.contiguous().view(-1, x2.size(2))
        x1_o = self.f(self.proj_1(x1_flat)).view(x1.size(0), x1.size(1), -1)
        # x2_o = self.f(self.proj_1(x2_flat)).view(x2.size(0), x2.size(1), -1)
        x2_o = self.f(self.proj_2(x2_flat)).view(x2.size(0), x2.size(1), -1)
        if self.scale_on:
            scalar = self.scalar.expand_as(x2_o)
            x2_o = scalar * x2_o
        scores = x1_o.bmm(x2_o.transpose(1, 2))
        return scores


class Bilinear(nn.Module):
    def __init__(self, x1_dim, x2_dim, prefix='sim', opt={}, dropout=None):
        super(Bilinear, self).__init__()
        self.opt = opt
        self.layer_norm_on = opt.get('{}_norm_on'.format(self.prefix), False)
        self.transform_on = opt.get('{}_proj_on'.format(self.prefix), False)
        # self.init = init_wrapper(opt.get('{}_init'.format(self.prefix), ''))
        self.dropout = dropout
        if self.transform_on:
            self.proj = nn.Linear(x1_dim, x2_dim)
            # self.init(self.proj.weight)
            if self.layer_norm_on: self.proj = weight_norm(self.proj)

    def forward(self, x, y):
        """
        x = batch * len * h1
        y = batch * h2
        x_mask = batch * len
        """
        if self.dropout:
            x = self.dropout(x)
            y = self.dropout(y)

        proj = self.proj(y) if self.transform_on else y
        if self.dropout:
            proj = self.dropout(proj)
        scores = x.bmm(proj.unsqueeze(2)).squeeze(2)
        return scores


class BilinearSum(nn.Module):
    def __init__(self, x1_dim, x2_dim, prefix='sim', opt={}, dropout=None):
        super(BilinearSum, self).__init__()
        self.x_linear = nn.Linear(x1_dim, 1, bias=False)
        self.y_linear = nn.Linear(x2_dim, 1, bias=False)
        self.layer_norm_on = opt.get('{}_norm_on'.format(self.prefix), False)
        self.init = init_wrapper(opt.get('{}_init'.format(self.prefix), False))
        if self.layer_norm_on:
            self.x_linear = weight_norm(self.x_linear)
            self.y_linear = weight_norm(self.y_linear)

        self.init(self.x_linear.weight)
        self.init(self.y_linear.weight)
        self.dropout = dropout

    def forward(self, x1, x2):
        """
        x1: batch * len1 * input_size
        x2: batch * len2 * input_size
        score: batch * len1 * len2
        """
        if self.dropout:
            x1 = self.dropout(x1)
            x2 = self.dropout(x2)

        x1_logits = self.x_linear(x1.contiguous().view(-1, x1.size(-1))).view(x1.size(0), -1, 1)
        x2_logits = self.y_linear(x2.contiguous().view(-1, x2.size(-1))).view(x2.size(0), 1, -1)

        shape = (x1.size(0), x1.size(1), x2.size())
        scores = x1_logits.expand_as(shape) + x2_logits.expand_as(shape)
        return scores


class Trilinear(nn.Module):
    """Function used in BiDAF"""
    def __init__(self, x1_dim, x2_dim, prefix='sim', opt={}, dropout=None):
        super(Trilinear, self).__init__()
        self.prefix = prefix
        self.x_linear = nn.Linear(x1_dim, 1, bias=False)
        self.x_dot_linear = nn.Linear(x1_dim, 1, bias=False)
        self.y_linear = nn.Linear(x2_dim, 1, bias=False)
        self.layer_norm_on = opt.get('{}_norm_on'.format(self.prefix), False)
        self.init = init_wrapper(opt.get('{}_init'.format(self.prefix), 'xavier_uniform'))
        if self.layer_norm_on:
            self.x_linear = weight_norm(self.x_linear)
            self.x_dot_linear = weight_norm(self.x_dot_linear)
            self.y_linear = weight_norm(self.y_linear)

        self.init(self.x_linear.weight)
        self.init(self.x_dot_linear.weight)
        self.init(self.y_linear.weight)
        self.dropout = dropout

    def forward(self, x1, x2):
        """
        x1: batch * len1 * input_size
        x2: batch * len2 * input_size
        score: batch * len1 * len2
        """
        if self.dropout:
            x1 = self.dropout(x1)
            x2 = self.dropout(x2)

        x1_logits = self.x_linear(x1.contiguous().view(-1, x1.size(-1))).view(x1.size(0), -1, 1)
        x2_logits = self.y_linear(x2.contiguous().view(-1, x2.size(-1))).view(x2.size(0), 1, -1)
        x1_dot = self.x_dot_linear(x1.contiguous().view(-1, x1.size(-1))).view(x1.size(0), -1, 1).expand_as(x1)
        x1_dot = x1 * x1_dot

        scores = x1_dot.bmm(x2.transpose(1, 2))
        scores += x1_logits.expand_as(scores) + x2_logits.expand_as(scores)
        return scores


class SimilarityWrapper(nn.Module):
    def __init__(self, x1_dim, x2_dim, prefix='attention', opt={}, dropout=None):
        super(SimilarityWrapper, self).__init__()
        self.score_func_str = opt.get('{}_sim_func'.format(prefix), 'dotproductproject').lower()
        self.score_func = None
        if self.score_func_str == 'dotproduct':
            self.score_func = DotProduct(x1_dim, x2_dim, prefix=prefix, opt=opt, dropout=dropout)
        elif self.score_func_str == 'dotproductproject':
            self.score_func = DotProductProject(x1_dim, x2_dim, prefix=prefix, opt=opt, dropout=dropout)
        elif self.score_func_str == 'bilinear':
            self.score_func = Bilinear(x1_dim, x2_dim, prefix=prefix, opt=opt, dropout=dropout)
        elif self.score_func_str == 'bilinearsum':
            self.score_func = BilinearSum(x1_dim, x2_dim, prefix=prefix, opt=opt, dropout=dropout)
        elif self.score_func_str == 'trilinear':
            self.score_func = Trilinear(x1_dim, x2_dim, prefix=prefix, opt=opt, dropout=dropout)
        else:
            raise NotImplementedError

    def forward(self, x1, x2):
        scores = self.score_func(x1, x2)
        return scores


class AttentionWrapper(nn.Module):
    def __init__(self, x1_dim, x2_dim, x3_dim=None, prefix='attention', opt={}, dropout=None):
        super(AttentionWrapper, self).__init__()
        self.prefix = prefix
        self.att_dropout = opt.get('{}_att_dropout'.format(self.prefix), 0)
        self.score_func = SimilarityWrapper(x1_dim, x2_dim, prefix=prefix, opt=opt, dropout=dropout)
        self.drop_diagonal = opt.get('{}_drop_diagonal'.format(self.prefix), False)
        self.output_size = x2_dim if x3_dim is None else x3_dim

    def forward(self, query, key, value, key_padding_mask=None, return_scores=False):
        logits = self.score_func(query, key)
        key_mask = key_padding_mask.unsqueeze(1).expand_as(logits)
        logits.data.masked_fill_(key_mask.data, -float('inf'))
        if self.drop_diagonal:
            assert logits.size(1) == logits.size(2)
            diag_mask = torch.diag(logits.data.new(logits.size(1)).zero_() + 1).byte().unsqueeze(0).expand_as(logits)
            logits.data.masked_fill_(diag_mask, -float('inf'))

        prob = F.softmax(logits.view(-1, key.size(1)), 1)
        prob = prob.view(-1, query.size(1), key.size(1))
        if self.att_dropout > 0:
            prob = self.dropout(prob)

        if value is None:
            value = key
        attn = prob.bmm(value)
        if return_scores:
            return attn, prob, logits
        else:
            return attn


class LinearSelfAttn(nn.Module):
    """Self attention over a sequence:
    * o_i = softmax(Wx_i) for x_i in X.
    """
    def __init__(self, input_size, dropout=None):
        super(LinearSelfAttn, self).__init__()
        self.linear = nn.Linear(input_size, 1)
        self.dropout = dropout

    def forward(self, x, x_mask):
        x = self.dropout(x)
        x_flat = x.contiguous().view(-1, x.size(-1))
        scores = self.linear(x_flat).view(x.size(0), x.size(1))
        scores.data.masked_fill_(x_mask.data, -float('inf'))
        alpha = F.softmax(scores, 1)
        return alpha.unsqueeze(1).bmm(x).squeeze(1)


class MLPSelfAttn(nn.Module):
    def __init__(self, input_size, opt={}, prefix='attn_sum', dropout=None):
        super(MLPSelfAttn, self).__init__()
        self.prefix = prefix
        self.FC = nn.Linear(input_size, input_size)
        self.linear = nn.Linear(input_size, 1)
        self.layer_norm_on = opt.get('{}_norm_on'.format(self.prefix), False)
        self.f = activation(opt.get('{}_activation'.format(self.prefix), 'relu'))
        if dropout is None:
            self.dropout = DropoutWrapper(opt.get('{}_dropout_p'.format(self.prefix), 0))
        else:
            self.dropout = dropout
        if self.layer_norm_on:
            self.FC = weight_norm(self.FC)

    def forward(self, x, x_mask):
        x = self.dropout(x)
        x_flat = x.contiguous().view(-1, x.size(-1))
        scores = self.linear(self.f(self.FC(x_flat))).view(x.size(0), x.size(1))
        scores.data.masked_fill_(x_mask.data, -float('inf'))
        alpha = F.softmax(scores)
        return alpha.unsqueeze(1).bmm(x).squeeze(1)


class SelfAttnWrapper(nn.Module):
    def __init__(self, input_size, prefix='attn_sum', opt={}, dropout=None):
        super(SelfAttnWrapper, self).__init__()
        """
        Self att wrapper, support linear and MLP
        """
        attn_type = opt.get('{}_type'.format(prefix), 'linear')
        if attn_type == 'mlp':
            self.att = MLPSelfAttn(input_size, prefix, opt, dropout)
        else:
            self.att = LinearSelfAttn(input_size, dropout)

    def forward(self, x, x_mask):
        return self.att(x, x_mask)


class DeepAttentionWrapper(nn.Module):
    def __init__(self, x1_dim, x2_dim, x3_dims, att_cnt, prefix='deep_att', opt=None, dropout=None):
        super(DeepAttentionWrapper, self).__init__()
        self.opt = {} if opt is None else opt
        self.prefix = prefix
        self.x1_dim = x1_dim
        self.x2_dim = x2_dim
        self.x3_dims = x3_dims

        if dropout is None:
            self.dropout = DropoutWrapper(opt.get('{}_dropout_p'.format(self.prefix), 0))
        else:
            self.dropout = dropout

        self.attn_list = nn.ModuleList()
        for i in range(0, att_cnt):
            if opt['multihead_on']:
                attention = MultiheadAttentionWrapper(self.x1_dim, self.x2_dim, self.x3_dims[i], prefix, opt, dropout=dropout)
            else:
                attention = AttentionWrapper(self.x1_dim, self.x2_dim, self.x3_dims[i], prefix, opt, self.dropout)
            self.attn_list.append(attention)

    def forward(self, x1, x2, x3, x2_mask):
        rvl = []
        for i in range(0, len(x3)):
            hiddens = self.attn_list[i](x1, x2, x3[i], x2_mask)
            rvl.append(hiddens)

        return torch.cat(rvl, 2)


class BilinearFlatSim(nn.Module):
    """A bilinear attention layer over a sequence X w.r.t y:
    * o_i = x_i'Wy for x_i in X.
    """
    def __init__(self, x_size, y_size, opt={}, prefix='seqatt', dropout=None):
        super(BilinearFlatSim, self).__init__()
        self.opt = opt
        self.weight_norm_on = opt.get('{}_weight_norm_on'.format(prefix), False)
        self.linear = nn.Linear(y_size, x_size)
        if self.weight_norm_on:
            self.linear = weight_norm(self.linear)
        if dropout is None:
            self.dropout = DropoutWrapper(opt.get('{}_dropout_p'.format(self.prefix), 0))
        else:
            self.dropout = dropout

    def forward(self, x, y, x_mask):
        """
        x = batch * len * h1
        y = batch * h2
        x_mask = batch * len
        """
        x = self.dropout(x)
        y = self.dropout(y)

        Wy = self.linear(y)
        xWy = x.bmm(Wy.unsqueeze(2)).squeeze(2)
        xWy.data.masked_fill_(x_mask.data, -float('inf'))
        return xWy


class SimpleFlatSim(nn.Module):
    def __init__(self, x_size, y_size, opt={}, prefix='seqatt', dropout=None):
        super(SimpleFlatSim, self).__init__()
        self.opt = opt
        self.weight_norm_on = opt.get('{}_norm_on'.format(prefix), False)
        self.linear = nn.Linear(y_size + x_size, 1)
        if self.weight_norm_on:
            self.linear = weight_norm(self.linear)
        if dropout is None:
            self.dropout = DropoutWrapper(opt.get('{}_dropout_p'.format(self.prefix), 0))
        else:
            self.dropout = dropout

    def forward(self, x, y, x_mask):
        """
        x = batch * len * h1
        y = batch * h2
        x_mask = batch * len
        """
        x = self.dropout(x)
        y = self.dropout(y)
        y = y.unsqueeze(1).expand_as(x)
        flat_x = torch.cat([x, y], 2).contiguous().view(x.size(0) * x.size(1), -1)
        flat_scores = self.linear(flat_x)
        scores = flat_scores.contiguous().view(x.size(0), -1)
        scores.data.masked_fill_(x_mask.data, -float('inf'))
        return scores


class FlatSim(nn.Module):
    def __init__(self, x_size, y_size, opt={}, prefix='seqatt', dropout=None):
        super(FlatSim, self).__init__()
        assert x_size == y_size
        self.opt = opt
        self.weight_norm_on = opt.get('{}_weight_norm_on'.format(prefix), False)
        self.linear = nn.Linear(x_size * 3, 1)
        if self.weight_norm_on:
            self.linear = weight_norm(self.linear)
        if dropout is None:
            self.dropout = DropoutWrapper(opt.get('{}_dropout_p'.format(self.prefix), 0))
        else:
            self.dropout = dropout

    def forward(self, x, y, x_mask):
        """
        x = batch * len * h1
        y = batch * h2
        x_mask = batch * len
        """
        x = self.dropout(x)
        y = self.dropout(y)
        y = y.unsqueeze(1).expand_as(x)

        flat_x = torch.cat([x, y, x * y], 2).contiguous().view(x.size(0) * x.size(1), -1)
        flat_scores = self.linear(flat_x)
        scores = flat_scores.contiguous().view(x.size(0), -1)
        scores.data.masked_fill_(x_mask.data, -float('inf'))

        return scores


class FlatSimV2(nn.Module):
    def __init__(self, x_size, y_size, opt={}, prefix='seqatt', dropout=None):
        super(FlatSimV2, self).__init__()
        assert x_size == y_size
        self.opt = opt
        self.weight_norm_on = opt.get('{}_weight_norm_on'.format(prefix), False)
        self.linear = nn.Linear(x_size * 4, 1)
        if self.weight_norm_on:
            self.linear = weight_norm(self.linear)
        if dropout is None:
            self.dropout = DropoutWrapper(opt.get('{}_dropout_p'.format(self.prefix), 0))
        else:
            self.dropout = dropout

    def forward(self, x, y, x_mask):
        """
        x = batch * len * h1
        y = batch * h2
        x_mask = batch * len
        """
        x = self.dropout(x)
        y = self.dropout(y)
        y = y.unsqueeze(1).expand_as(x)

        flat_x = torch.cat([x, y, x * y, torch.abs(x - y)], 2).contiguous().view(x.size(0) * x.size(1), -1)
        flat_scores = self.linear(flat_x)
        scores = flat_scores.contiguous().view(x.size(0), -1)
        scores.data.masked_fill_(x_mask.data, -float('inf'))

        return scores


class FlatSimilarityWrapper(nn.Module):
    def __init__(self, x1_dim, x2_dim, prefix='attention', opt={}, dropout=None):
        super(FlatSimilarityWrapper, self).__init__()
        self.score_func_str = opt.get('{}_att_type'.format(prefix), 'none').lower()
        self.att_dropout = DropoutWrapper(opt.get('{}_att_dropout'.format(prefix), 0))
        self.score_func = None
        if self.score_func_str == 'bilinear':
            self.score_func = BilinearFlatSim(x1_dim, x2_dim, prefix=prefix, opt=opt, dropout=dropout)
        elif self.score_func_str == 'simple':
            self.score_func = SimpleFlatSim(x1_dim, x2_dim, prefix=prefix, opt=opt, dropout=dropout)
        elif self.score_func_str == 'flatsim':
            self.score_func = FlatSim(x1_dim, x2_dim, prefix=prefix, opt=opt, dropout=dropout)
        else:
            self.score_func = FlatSimV2(x1_dim, x2_dim, prefix=prefix, opt=opt, dropout=dropout)

    def forward(self, x1, x2, mask):
        scores = self.score_func(x1, x2, mask)
        return scores

class MultiheadAttentionWrapper(nn.Module):
    """Multi-headed attention.
    See "Attention Is All You Need" for more details.
    """
    def __init__(self, query_dim, key_dim, value_dim, prefix='attention', opt={}, dropout=None):
        super().__init__()
        self.prefix = prefix

        self.num_heads = opt.get('{}_head'.format(self.prefix), 1)
        self.dropout = DropoutWrapper(opt.get('{}_dropout'.format(self.prefix), 0)) if dropout is None else dropout

        self.qkv_dim = [query_dim, key_dim, value_dim]
        assert query_dim == key_dim, "query dim must equal with key dim"

        self.hidden_size = opt.get('{}_hidden_size'.format(self.prefix), 64)

        self.proj_on = opt.get('{}_proj_on'.format(prefix), False)
        self.share = opt.get('{}_share'.format(self.prefix), False)
        self.layer_norm_on = opt.get('{}_norm_on'.format(self.prefix), False)
        self.scale_on = opt.get('{}_scale_on'.format(self.prefix), False)

        if self.proj_on:
            self.proj_modules = nn.ModuleList([nn.Linear(dim, self.hidden_size) for dim in self.qkv_dim[0:2]])
            if self.layer_norm_on:
                for proj in self.proj_modules:
                    proj = weight_norm(proj)
            if self.share and self.qkv_dim[0] == self.qkv_dim[1]:
                self.proj_modules[1] = self.proj_modules[0]
            self.f = activation(opt.get('{}_activation'.format(self.prefix), 'relu'))

            self.qkv_head_dim = [self.hidden_size // self.num_heads] * 3
            self.qkv_head_dim[2] = value_dim // self.num_heads
            assert self.qkv_head_dim[0] * self.num_heads == self.hidden_size, "hidden size must be divisible by num_heads"
            assert self.qkv_head_dim[2] * self.num_heads == value_dim, "value size must be divisible by num_heads"

        else:
            self.qkv_head_dim = [emb // self.num_heads for emb in self.qkv_dim]
            #import pdb; pdb.set_trace()
            assert self.qkv_head_dim[0] * self.num_heads == self.qkv_dim[0], "query size must be divisible by num_heads"
            assert self.qkv_head_dim[1] * self.num_heads == self.qkv_dim[1], "key size must be divisible by num_heads"
            assert self.qkv_head_dim[2] * self.num_heads == self.qkv_dim[2], "value size must be divisible by num_heads"

        if self.scale_on:
            self.scaling = self.qkv_head_dim[0]**-0.5
        self.drop_diagonal = opt.get('{}_drop_diagonal'.format(self.prefix), False)
        self.output_size = self.qkv_dim[2]

    def forward(self, query, key, value, key_padding_mask=None):
        query = query.transpose(0, 1)
        key = key.transpose(0, 1)
        value = value.transpose(0, 1)

        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.qkv_dim[0]

        q, k, v = query, key, value
        if self.proj_on:
            if self.dropout:
                q, k = self.dropout(q), self.dropout(k)
            q, k = [self.f(proj(input)) for input, proj in zip([query, key], self.proj_modules)]

        src_len = k.size(0)
        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        if self.scale_on:
            q *= self.scaling

        q = q.contiguous().view(tgt_len, bsz*self.num_heads, self.qkv_head_dim[0]).transpose(0, 1)
        k = k.contiguous().view(src_len, bsz*self.num_heads, self.qkv_head_dim[1]).transpose(0, 1)
        v = v.contiguous().view(src_len, bsz*self.num_heads, self.qkv_head_dim[2]).transpose(0, 1)

        attn_weights = torch.bmm(q, k.transpose(1, 2))
        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        if key_padding_mask is not None:
            # don't attend to padding symbols
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.float().masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf'),
            ).type_as(attn_weights)  # FP16 support: cast to float and back
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if self.drop_diagonal:
            assert attn_weights.size(1) == attn_weights.size(2)
            diag_mask = torch.diag(attn_weights.data.new(attn_weights.size(1)).zero_() + 1).byte().unsqueeze(0).expand_as(attn_weights)
            attn_weights.data.masked_fill_(diag_mask, -float('inf'))

        attn_weights = F.softmax(attn_weights.float(), dim=-1).type_as(attn_weights)
        attn_weights = self.dropout(attn_weights)

        attn = torch.bmm(attn_weights, v)
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.qkv_head_dim[2]]
        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, -1)

        # output_shape: Batch * Time * Channel
        attn = attn.transpose(0, 1)

        return attn
