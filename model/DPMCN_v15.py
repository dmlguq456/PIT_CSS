#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Implementation of Conformer speech separation model"""

import math
import numpy
import torch
from torch import nn
from torch.autograd import Variable


class GlobalLayerNorm(nn.Module):
    '''
       Calculate Global Layer Normalization
       dim: (int or list or torch.Size) –
          input shape from an expected input of size
       eps: a value added to the denominator for numerical stability.
       elementwise_affine: a boolean value that when set to True, 
          this module has learnable per-element affine parameters 
          initialized to ones (for weights) and zeros (for biases).
    '''

    def __init__(self, dim, shape, eps=1e-8, elementwise_affine=True):
        super(GlobalLayerNorm, self).__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            if shape == 3:
                self.weight = nn.Parameter(torch.ones(self.dim, 1))
                self.bias = nn.Parameter(torch.zeros(self.dim, 1))
            if shape == 4:
                self.weight = nn.Parameter(torch.ones(self.dim, 1, 1))
                self.bias = nn.Parameter(torch.zeros(self.dim, 1, 1))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x):
        # x = N x C x K x S or N x C x L
        # N x 1 x 1
        # cln: mean,var N x 1 x K x S
        # gln: mean,var N x 1 x 1
        if x.dim() == 4:
            mean = torch.mean(x, (1, 2, 3), keepdim=True)
            var = torch.mean((x-mean)**2, (1, 2, 3), keepdim=True)
            if self.elementwise_affine:
                x = self.weight*(x-mean)/torch.sqrt(var+self.eps)+self.bias
            else:
                x = (x-mean)/torch.sqrt(var+self.eps)
        if x.dim() == 3:
            mean = torch.mean(x, (1, 2), keepdim=True)
            var = torch.mean((x-mean)**2, (1, 2), keepdim=True)
            if self.elementwise_affine:
                x = self.weight*(x-mean)/torch.sqrt(var+self.eps)+self.bias
            else:
                x = (x-mean)/torch.sqrt(var+self.eps)
        return x


class RelativePositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, maxlen=8000, embed_v=False):
        super(RelativePositionalEncoding, self).__init__()

        self.d_model = d_model
        self.maxlen = maxlen
        self.pe_k = torch.nn.Embedding(2*maxlen, d_model)
        if embed_v:
            self.pe_v = torch.nn.Embedding(2*maxlen, d_model)
        self.embed_v = embed_v

    def forward(self, pos_seq):
        pos_seq.clamp_(-self.maxlen, self.maxlen - 1)
        pos_seq = pos_seq + self.maxlen
        if self.embed_v:
            return self.pe_k(pos_seq), self.pe_v(pos_seq)
        else:
            return self.pe_k(pos_seq), None


class MultiHeadedAttention(nn.Module):
    """Multi-Head Attention layer.

    :param int n_head: the number of head s
    :param int n_feat: the number of features
    :param float dropout_rate: dropout rate

    """

    def __init__(self, n_head, n_feat, dropout_rate):
        """Construct an MultiHeadedAttention object."""
        super(MultiHeadedAttention, self).__init__()
        assert n_feat % n_head == 0
        # We assume d_v always equals d_k
        self.d_k = n_feat // n_head
        self.h = n_head
        self.layer_norm = nn.LayerNorm(n_feat)
        self.linear_q = nn.Linear(n_feat, n_feat)
        self.linear_k = nn.Linear(n_feat, n_feat)
        self.linear_v = nn.Linear(n_feat, n_feat)

        self.linear_out = nn.Linear(n_feat, n_feat)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x, pos_k, mask):
        """Compute 'Scaled Dot Product Attention'.

        :param torch.Tensor mask: (batch, time1, time2)
        :param torch.nn.Dropout dropout:
        :return torch.Tensor: attentined and transformed `value` (batch, time1, d_model)
             weighted by the query dot key attention (batch, head, time1, time2)
        """
        n_batch = x.size(0)
        x = self.layer_norm(x)
        q = self.linear_q(x).view(n_batch, -1, self.h, self.d_k)  #(b, t, d)
        k = self.linear_k(x).view(n_batch, -1, self.h, self.d_k)   #(b, t, d)
        v = self.linear_v(x).view(n_batch, -1, self.h, self.d_k)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)  # (batch, head, time2, d_k)
        v = v.transpose(1, 2)  # (batch, head, time2, d_k)
        A = torch.matmul(q, k.transpose(-2, -1))
        reshape_q = q.contiguous().view(n_batch * self.h, -1, self.d_k).transpose(0,1)
        if pos_k is not None:
            B = torch.matmul(reshape_q, pos_k.transpose(-2, -1))
            B = B.transpose(0, 1).view(n_batch, self.h, pos_k.size(0), pos_k.size(1))
            scores = (A + B) / math.sqrt(self.d_k)
        else:
            scores = A / math.sqrt(self.d_k)
        if mask is not None:
            mask = mask.unsqueeze(1).eq(0)  # (batch, 1, time1, time2)
            min_value = float(numpy.finfo(torch.tensor(0, dtype=scores.dtype).numpy().dtype).min)
            scores = scores.masked_fill(mask, min_value)
            self.attn = torch.softmax(scores, dim=-1).masked_fill(mask, 0.0)  # (batch, head, time1, time2)
        else:
            self.attn = torch.softmax(scores, dim=-1)  # (batch, head, time1, time2)

        p_attn = self.dropout(self.attn)
        x = torch.matmul(p_attn, v)  # (batch, head, time1, d_k)
        x = x.transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k)  # (batch, time1, d_model)
        return self.dropout(self.linear_out(x))  # (batch, time1, d_model)

# class GRN(nn.Module):
#     """ GRN (Global Response Normalization) layer
#     """
#     def __init__(self, dim):
#         super().__init__()
#         self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
#         self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

#     def forward(self, x):
#         Gx = torch.norm(x, p=2, dim=2, keepdim=True)
#         Gx = Gx.mean(dim=1, keepdim=True)
#         Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
#         return self.gamma * (x * Nx) + self.beta + x


class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer
    """
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1,2), keepdim=True)
        Gx = Gx / x.shape[1]
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


class Cross_MultiHeadedAttention(nn.Module):
    """Multi-Head Attention layer.

    :param int n_head: the number of head s
    :param int n_feat: the number of features
    :param float dropout_rate: dropout rate

    """

    def __init__(self, n_head, n_feat, dropout_rate):
        """Construct an MultiHeadedAttention object."""
        super(Cross_MultiHeadedAttention, self).__init__()
        assert n_feat % n_head == 0
        # We assume d_v always equals d_k
        self.d_k = n_feat // n_head
        self.h = n_head
        self.layer_norm = nn.LayerNorm(n_feat)
        self.linear_q = nn.Linear(n_feat, n_feat)
        self.linear_k = nn.Linear(n_feat, n_feat)
        self.linear_v = nn.Linear(n_feat, n_feat)

        self.linear_out = nn.Linear(n_feat, n_feat)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x, q_in, pos_k, mask):
        """Compute 'Scaled Dot Product Attention'.

        :param torch.Tensor mask: (batch, time1, time2)
        :param torch.nn.Dropout dropout:
        :return torch.Tensor: attentined and transformed `value` (batch, time1, d_model)
             weighted by the query dot key attention (batch, head, time1, time2)
        """
        n_batch = x.size(0)
        x = self.layer_norm(x)
        q = self.linear_q(q_in).view(n_batch, -1, self.h, self.d_k)  #(b, t, d)
        k = self.linear_k(x).view(n_batch, -1, self.h, self.d_k)   #(b, t, d)
        v = self.linear_v(x).view(n_batch, -1, self.h, self.d_k)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)  # (batch, head, time2, d_k)
        v = v.transpose(1, 2)  # (batch, head, time2, d_k)
        A = torch.matmul(q, k.transpose(-2, -1))
        reshape_q = q.contiguous().view(n_batch * self.h, -1, self.d_k).transpose(0,1)
        if pos_k is not None:
            B = torch.matmul(reshape_q, pos_k.transpose(-2, -1))
            B = B.transpose(0, 1).view(n_batch, self.h, pos_k.size(0), pos_k.size(1))
            scores = (A + B) / math.sqrt(self.d_k)
        else:
            scores = A / math.sqrt(self.d_k)
        if mask is not None:
            mask = mask.unsqueeze(1).eq(0)  # (batch, 1, time1, time2)
            min_value = float(numpy.finfo(torch.tensor(0, dtype=scores.dtype).numpy().dtype).min)
            scores = scores.masked_fill(mask, min_value)
            self.attn = torch.softmax(scores, dim=-1).masked_fill(mask, 0.0)  # (batch, head, time1, time2)
        else:
            self.attn = torch.softmax(scores, dim=-1)  # (batch, head, time1, time2)

        p_attn = self.dropout(self.attn)
        x = torch.matmul(p_attn, v)  # (batch, head, time1, d_k)
        x = x.transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k)  # (batch, time1, d_model)
        return self.dropout(self.linear_out(x))  # (batch, time1, d_model)





class ConvModule(nn.Module):
    def __init__(self, input_dim, kernel_size, dropout_rate, causal=False):
        super(ConvModule, self).__init__()
        self.layer_norm = nn.LayerNorm(input_dim)

        self.pw_conv_1 = nn.Linear(input_dim, 2*input_dim)
        self.input_dim = input_dim
        self.glu_act = torch.nn.Sigmoid()
        self.causal = causal
        if causal:
            self.dw_conv_1d = nn.Conv1d(input_dim, input_dim, kernel_size, 1, padding=(kernel_size-1), groups=input_dim)
        else:
            self.dw_conv_1d = nn.Conv1d(input_dim, input_dim, kernel_size, 1, padding=(kernel_size-1)//2, groups=input_dim)
        self.BN = nn.BatchNorm1d(input_dim)
        #todo ReLU() --> Swish()
        self.act = nn.ReLU()
        self.pw_conv_2 = nn.Linear(input_dim, input_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.kernel_size = kernel_size

    def forward(self, x):
        x = self.layer_norm(x)
        x = self.pw_conv_1(x)
        x = x[...,self.input_dim:] * self.glu_act(x[...,:self.input_dim])
        x = x.permute([0, 2, 1])
        x = self.dw_conv_1d(x)
        if self.causal:
            x = x[:, :, :-(self.kernel_size-1)]
        x = self.BN(x)
        x = self.act(x)
        x = x.permute([0, 2, 1])
        x = self.pw_conv_2(x)
        # x = self.pw_conv_3(x)
        x = self.dropout(x)
        return x
    
class FeedForward(nn.Module):
    def __init__(self, d_model, d_inner, dropout_rate):
        super(FeedForward, self).__init__()

        self.d_model = d_model
        self.d_inner = d_inner

        self.layer_norm = nn.LayerNorm(d_model)
        self.net = nn.Sequential(
            nn.Linear(d_model, d_inner),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(d_inner, d_model),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x):
        x = self.layer_norm(x)
        out = self.net(x)

        return out


class Dual_intra_block(nn.Module):

    def __init__(self, d_model, n_head, d_ffn, kernel_size, dropout_rate, causal=False):
        """Construct an EncoderLayer object."""
        super(Dual_intra_block, self).__init__()

        self.intra_feed_forward_in = FeedForward(d_model, d_ffn, dropout_rate)
        self.intra_self_attn = MultiHeadedAttention(n_head, d_model, dropout_rate)
        self.intra_conv = ConvModule(d_model, kernel_size, dropout_rate, causal=causal)
        self.intra_feed_forward_out = FeedForward(d_model, d_ffn, dropout_rate)
        self.intra_layer_norm = nn.LayerNorm(d_model)

        self.pos_emb = RelativePositionalEncoding(d_model // n_head, 2000, False)


    def forward(self, x, mask):

        B, T, C, F = x.shape
        # [ B , F , K , S ]        

        intra_x = x.permute(0, 2, 1, 3).contiguous().view(B*C, T, F).contiguous()

        intra_x = intra_x + 0.5 * self.intra_feed_forward_in(intra_x)

        pos_seq = torch.arange(0, T).long().to(intra_x.device)
        pos_seq = pos_seq[:, None] - pos_seq[None, :]
        pos_k_intra, _ = self.pos_emb(pos_seq)

        intra_x = intra_x + self.intra_self_attn(intra_x, pos_k_intra, mask)
        intra_x = intra_x + self.intra_conv(intra_x)
        intra_x = intra_x + 0.5 * self.intra_feed_forward_out(intra_x)
        intra_x = self.intra_layer_norm(intra_x)

        intra_x = intra_x.view(B, C, T, F).contiguous()
        intra_x = intra_x.permute(0, 2, 1, 3).contiguous()
        # [ B, T, C, F ]
        return intra_x

class Dual_inter_block(nn.Module):

    def __init__(self, d_model, n_head, d_ffn, kernel_size, dropout_rate, causal=False):
        """Construct an EncoderLayer object."""
        super(Dual_inter_block, self).__init__()

        self.inter_feed_forward_in = FeedForward(d_model, d_ffn, dropout_rate)
        self.inter_self_attn = MultiHeadedAttention(n_head, d_model, dropout_rate)
        self.inter_conv = ConvModule(d_model, kernel_size, dropout_rate, causal=causal)
        self.inter_feed_forward_out = FeedForward(d_model, d_ffn, dropout_rate)
        self.inter_layer_norm = nn.LayerNorm(d_model)

        self.pos_emb = RelativePositionalEncoding(d_model // n_head, 2000, False)


    def forward(self, x, mask):
        
        B, T, C, F = x.shape

        inter_x = x.permute(0, 3, 1, 2).contiguous().view(B*F, T, C).contiguous()

        inter_x = inter_x + 0.5 * self.inter_feed_forward_in(inter_x)

        pos_seq = torch.arange(0, T).long().to(inter_x.device)
        pos_seq = pos_seq[:, None] - pos_seq[None, :]
        pos_k_inter, _ = self.pos_emb(pos_seq)

        inter_x = inter_x + self.inter_self_attn(inter_x, pos_k_inter, mask)
        inter_x = inter_x + self.inter_conv(inter_x)
        inter_x = inter_x + 0.5 * self.inter_feed_forward_out(inter_x)
        inter_x = self.inter_layer_norm(inter_x)

        inter_x = inter_x.view(B, F, T, C).contiguous()
        inter_x = inter_x.permute(0, 2, 3, 1).contiguous()
        # [B, T , C, F]
        return inter_x

class Dual_Path_Layer(nn.Module):

    def __init__(self, d_model_F, d_model_C, n_head_F, n_head_C, d_ffn_F, d_ffn_C, kernel_size, dropout_rate, causal=False, N_intra=1, N_inter=1, GRN_opt=True):
        """Construct an EncoderLayer object."""
        super(Dual_Path_Layer, self).__init__()

        self.intra_block = nn.Sequential(
            *[Dual_intra_block(d_model_F, n_head_F, d_ffn_F, kernel_size, dropout_rate, causal=causal) for _ in range(N_intra)]
            )        
        self.inter_block = nn.Sequential(
            *[Dual_inter_block(d_model_C, n_head_C, d_ffn_C, kernel_size, dropout_rate, causal=causal) for _ in range(N_inter)]
            )
        self.GRN_opt = GRN_opt
        if GRN_opt:
            self.GRN_C = GRN(d_model_C)
            self.GRN_F = GRN(d_model_F)

    def forward(self, x, mask):

        B, T, C, F = x.shape
        x_intra = x
        
        x_intra = x_intra.permute(0, 1, 3, 2).contiguous()
        if self.GRN_opt:
            x_intra = self.GRN_C(x_intra)
        x_intra = x_intra.view(B*T, F, C).contiguous()
        x_intra = torch.nn.functional.adaptive_avg_pool1d(x_intra, C//4)
        x_intra = x_intra.view(B, T, F, C//4).permute(0, 1, 3, 2).contiguous()
        for layer in self.intra_block:
            x_intra = layer(x_intra, mask)

        x_intra = x_intra.permute(0, 1, 3, 2).contiguous().view(B*T, F, C//4)
        x_intra = torch.nn.functional.upsample(x_intra, C)
        x_intra = x_intra.view(B, T, F, C).permute(0, 1, 3, 2).contiguous()

        x = x + x_intra

        x_inter = x
        if self.GRN_opt:
            x_inter = self.GRN_F(x_inter)
        x_inter = x_inter.view(B*T, C, F)
        x_inter = torch.nn.functional.adaptive_avg_pool1d(x_inter, F//4)
        x_inter = x_inter.view(B, T, C, F//4)

        for layer in self.inter_block:
            x_inter = layer(x_inter, mask)

        x_inter = x_inter.view(B*T, C, F//4)
        x_inter = torch.nn.functional.upsample(x_inter, F)
        x_inter = x_inter.view(B, T, C, F)
        x = x + x_inter
        
        return x

class FrequenceyWiseLinear(nn.Module):
    def __init__(self, num_bins, num_ch):
        super().__init__()
        self.freq_linear = nn.Parameter(torch.rand(1, num_bins, num_ch, num_ch))
  
    def forward(self, x):
        return torch.einsum("...fc,...fce->...fe",[x, self.freq_linear])

   
class AttentiveChannelPool(nn.Module):
    def __init__(self, num_bins, num_ch, reduct_ratio):
        super().__init__()
        
        self.freq_ffn_1 = nn.Linear(num_bins, num_bins//reduct_ratio)
        self.relu = nn.ReLU()
        self.ch_ffn = FrequenceyWiseLinear(num_bins//reduct_ratio, num_ch)
        self.freq_ffn_2 = nn.Linear(num_bins//reduct_ratio, num_bins)
        self.ch_softmax = nn.Softmax(dim=-1)
        self.layer_scale = nn.Parameter(torch.ones(1,1,num_bins), requires_grad=True)
        
    def forward(self, x):
        '''
        input : B x T x F x C
        '''
        y = x.mean(1)
        y = y.permute(0, 2, 1).contiguous()
        y = self.freq_ffn_1(y)
        y = self.relu(y)

        y = y.permute(0, 2, 1).contiguous()
        y = self.ch_ffn(y)
        y = y.permute(0, 2, 1).contiguous()

        y = self.freq_ffn_2(y)
        y = y.permute(0, 2, 1).contiguous()
        y = self.ch_softmax(y)
        
        y = x * y.unsqueeze(1)
        
        return y.sum(-1) * self.layer_scale # B, T, F




class DPConformer(nn.Module):
    """Conformer Encoder https://arxiv.org/abs/2005.08100
        """
    def __init__(self,
                 num_mic=7,
                 num_spk=2,
                 idim=257,
                 attention_dim_F=64,
                 attention_dim_C=32,
                 attention_heads_F=4,
                 attention_heads_C=2,
                 linear_units_F=256,
                 linear_units_C=128,
                 kernel_size=33,
                 dropout_rate=0.1,
                 causal=False,
                 N_intra=1,
                 N_inter=1,
                 N_repeat=4,
                 GRN_opt=True,
                 beta='vector'
                 ):
        super(DPConformer, self).__init__()
        if beta == 'vector':
            print('vector')
            self.exponent = nn.Parameter(torch.ones(idim,1), requires_grad=True)
        elif beta == 'scalar':
            print('scalar')
            self.exponent = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        elif beta == 'fixed_mag':
            print('fixed to 0.5')
            self.exponent = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        elif beta == 'fixed_no':
            print('no')
            self.exponent = nn.Parameter(torch.tensor(-1.0e10), requires_grad=False)

        self.IPD_factor = nn.Parameter(torch.ones(idim,1), requires_grad=True)
        self.sigmoid  = torch.nn.Sigmoid()
        self.embed_inter1 = torch.nn.Sequential(
            # SRP_weight(idim, num_mic, attention_dim_C*4),
            torch.nn.Linear(num_mic*(num_mic+1), attention_dim_C*4),
            torch.nn.Dropout(dropout_rate),
            torch.nn.ReLU()
        )
        self.embed_intra = torch.nn.Sequential(
            torch.nn.Linear(idim, attention_dim_F),
            torch.nn.Dropout(dropout_rate),
            torch.nn.ReLU()
        )
        self.embed_inter2 = torch.nn.Sequential(
            torch.nn.Linear(attention_dim_C*4, attention_dim_C),
            torch.nn.Dropout(dropout_rate),
            torch.nn.ReLU()
        )


        self.attention_dim_C = attention_dim_C

        self.layer_norm = nn.LayerNorm([attention_dim_C, attention_dim_F])
        self.relu = nn.ReLU()
    
        self.DP_conformer = torch.nn.Sequential(
            *[Dual_Path_Layer(attention_dim_F, attention_dim_C, attention_heads_F, attention_heads_C, linear_units_F, linear_units_C, kernel_size, dropout_rate, causal=causal, N_intra=N_intra, N_inter=N_inter, GRN_opt=GRN_opt)
            for _ in range(N_repeat)]
            )
        self.num_spk = num_spk
        self.mask_estim_num_spk = torch.nn.Linear(attention_dim_F, attention_dim_F*num_spk)
        self.mask_estim_spec = torch.nn.Conv1d(attention_dim_F, idim, kernel_size=1, padding=0)


        self.mask_estim_ch = torch.nn.Sequential(
            AttentiveChannelPool(idim, attention_dim_C, 4),
            torch.nn.Sigmoid()
        )



    def forward(self, x, masks):
        idim = x.shape[-1]//2
        B, T, idim, M = x.shape
        x_r = x[...,:M//2] # [B, T, 257, M]
        x_i = x[...,M//2:] # [B, T, 257, M]
        # x_mag = torch.sqrt(torch.square(x_r) + torch.square(x_i))
        xs = self.get_scm(x_r.permute(0, 3, 2, 1), x_i.permute(0, 3, 2, 1)) # [B, T, 257, M*M]
        xs_abs = xs.abs()
        beta_phat = torch.pow(xs_abs,self.sigmoid(self.exponent))
        xs_abs = xs_abs / beta_phat
        xs_angle = xs.angle() * self.sigmoid(self.IPD_factor)
        
        xs = torch.polar(xs_abs, xs_angle)

        xs = torch.view_as_real(xs)
        B, T, F, C, _ = xs.shape
        xs = xs.view(B, T, F, C*2).contiguous()

        # print(self.sigmoid_exp(self.exponent))
        # xs = torch.complex(x_r,x_i) # [B, T, 257, M]
        
        B, T, F, C = xs.shape

        xs = xs.permute(0, 2, 1, 3).contiguous().view(B*F, T, C)
        xs = self.embed_inter1(xs)
        # xs = torch.cat((self.embed_inter1(xs),xs[...,[0]].abs()),dim=-1)
        C = xs.shape[-1]
        xs = xs.view(B, F, T, C).contiguous()

        xs = xs.permute(0, 3, 2, 1).contiguous().view(B*C, T, F)
        xs = self.embed_intra(xs)
        F = xs.shape[-1]
        xs = xs.view(B, C, T, F).contiguous()

        xs = xs.permute(0, 3, 2, 1).contiguous().view(B*F, T, C)
        xs = self.embed_inter2(xs)
        C = xs.shape[-1]
        xs = xs.view(B, F, T, C).contiguous()


        xs = xs.permute(0, 2, 3, 1).contiguous()
        xs = self.layer_norm(xs)
        xs = self.relu(xs)
       # [ B, F, T ]
        for layer in self.DP_conformer:
            xs = layer(xs, masks)

        N = self.num_spk
        xs = xs.permute(0, 2, 1, 3).contiguous().view(B*C, T, F)
        xs = self.mask_estim_num_spk(xs)
        xs = xs.view(B, C, T, N, F).contiguous()
        xs = xs.permute(0, 3, 2, 4, 1).contiguous() # B, N, T, F, C
        xs = xs.view(B*N*T, F, C).contiguous()
        xs = self.mask_estim_spec(xs)
        F = xs.shape[-2]
        xs = xs.view(B*N, T, F, C).contiguous()
        # xs = xs.view(B, T, N, C, F).contiguous()
        # xs = xs.permute(0, 3, 1, 2, 4).contiguous()
        # xs = xs.view(B, C, T, self.num_spk, F).contiguous()

        #  B, T, F, C --> B, T, F, 1
        # xs = xs.permute(0, 2, 1).contiguous()
        xs = self.mask_estim_ch(xs)
        # C = xs.shape[-1]
        xs = xs.view(B, N, T, F).permute(0, 2, 1, 3).contiguous()
        xs = xs.view(B, T, N*F).contiguous()

        return xs


    def get_scm(self,
                x_r, # [... , ch , freq , time]
                x_i, # [... , ch , freq , time]
                mask=None # [... , freq , time]
                ): # [... , freq , ch*ch]

        """Compute weighted Spatial Covariance Matrix."""

        x = torch.complex(x_r,x_i)

        if mask is not None:
            x = x * torch.unsqueeze(torch.clamp(mask,min=1.0e-4),-3)

        x = x - torch.mean(x, dim=-1, keepdim=True)
        # pre-emphasis / normalize
        x_abs = x.abs()
        x_mean = torch.mean(x_abs**2, dim=-1, keepdim=True)
        x_norm = torch.sqrt(torch.sum(x_mean,dim=1, keepdim=True))
        x = x / x_norm

        x = torch.transpose(x, -3, -2)  # shape (batch, freq, ch, time)
        # outer product:
        # (..., ch_1, time) x (..., ch_2, time) -> (..., time, ch_1, ch_2)    
        scm = torch.einsum("...ct,...et->...tce", [x, x.conj()])
        B, F, T, C1, C2 = scm.shape
        idx = torch.triu_indices(C1,C2)
        scm = scm[...,idx[0],idx[1]]
        B, F, T, C = scm.shape

        scm = scm.permute(0, 2, 1, 3) # [B, T, F, C]
        return scm


default_encoder_conf = {
    "attention_dim_C": 32,
    "attention_dim_F": 64,
    "attention_heads_F": 4,
    "attention_heads_C": 2,
    "linear_units_F": 256,
    "linear_units_C": 128,
    "N_repeat": 8,
    "kernel_size": 33,
    "dropout_rate": 0.1,
    "relative_pos_emb": True
}



class DPMCN_v15(nn.Module):
    """
    Conformer speech separation model
    """
    def __init__(self,
                 stats_file=None,
                 in_features=257,
                 num_mics=7,
                 num_bins=257,
                 num_spks=2,
                 noise_flag=True,
                 crm=False,
                 IPD_sincos=False,
                 conformer_conf=default_encoder_conf):
        super().__init__()
        # input normalization layer
        self.crm = crm
        self.num_spks = num_spks
        num_nois = 1 if noise_flag else 0
        self.num_nois = num_nois
        self.noise_flag = noise_flag

        if IPD_sincos:
            self.num_mics = 2*num_mics - 1
        else:
            self.num_mics = num_mics
        
        if self.crm: self.num_mics += 1

        # Conformer Encoders
        self.conformer = DPConformer(self.num_mics, self.num_spks + self.num_nois, in_features, **conformer_conf)
        self.num_bins = num_bins
        self.in_features = in_features
        
    def forward(self, f, angle_dif=None, train=True):
        """
        args
            f: Batch x Time x Freq x 3Mic.
        return
            m: [Batch x Freq x Time, ...]
        """
        # N x * x T => N x T x *
        # f = f.transpose(1, 2)

        # global feature normalization
        if len(f.shape) == 2:
            f = f.unsqueeze(0)
        m = self.conformer(f, masks=None)

        m = m.transpose(1, 2)
        # B x T x F => B x F x T
        m = torch.chunk(m, self.num_spks + self.num_nois, 1)

        return m
