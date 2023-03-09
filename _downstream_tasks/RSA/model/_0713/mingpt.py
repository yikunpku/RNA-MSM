"""
taken from: https://github.com/karpathy/minGPT/
GPT model:
- the initial stem consists of a combination of token encoding and a positional encoding
- the meat of it is a uniform sequence of Transformer blocks
    - each Transformer is a sequential combination of a 1-hidden-layer MLP block and a self-attention block
    - all blocks feed into a central residual pathway similar to resnets
- the final decoder is a linear projection into a vanilla Softmax classifier
"""

import math
import logging

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F

logger = logging.getLogger(__name__)

class GPTConfig:
    """ base GPT config, params common to all GPT versions """
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    def __init__(self, vocab_size, block_size, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        for k,v in kwargs.items():
            setattr(self, k, v)


class SelfAttention(nn.Module):
    """
    A vanilla multi-head self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, n_embd,n_head,attn_pdrop=0.1,resid_pdrop=0.1):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        # output projection
        self.proj = nn.Linear(n_embd, n_embd)
        self.n_head = n_head

    def forward(self, x, mask=None):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        # k = torch.mul(x,mask)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        # k = torch.mul(x, mask)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        # k = torch.mul(x, mask)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        if mask is not None: # [16,1,340]
            t = mask.unsqueeze(3) # [16,1,340,1]
            t1 = t.repeat(1,att.size()[1],1,1) # [16,8,340,1]
            t2 = t1.transpose(-1, -2) # [16, 8, 1, 340]
            t3 = t1 @ t2  # ==== t1.mul(t2) [16, 8, 340, 340]
            att = att.masked_fill(t3[:,:,:T,:T] == 0, float('-inf')) # [16,8,340,340] [b,h,L,L]
        att = F.softmax(att, dim=-1) # att[0,0,0,:].sum()
        att = att.masked_fill(att.isnan(), 0.)  # [16,8,340,340] [b,h,L,L]
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y


class Block(nn.Module):
    """ an unassuming Transformer block """
    def __init__(self, num_features,*args,**kwargs):
        super().__init__()
        self.ln1 = nn.LayerNorm(num_features)
        self.ln2 = nn.LayerNorm(num_features)
        self.attn = SelfAttention(num_features,8,attn_pdrop=0.1,resid_pdrop=0.1)
        self.mlp = nn.Sequential(
            nn.Linear(num_features, 4 * num_features),
            nn.GELU(),  # nice
            nn.Linear(4 * num_features, num_features),
            nn.Dropout(0.1),
        )

    def forward(self, x,mask=None): # [16, 128, 340]
        x = x.transpose(2,1)
        x = x + self.attn(self.ln1(x),mask=mask) # [16, 340, 128]
        x = x.transpose(2,1)
        x = x.mul(mask)
        x = x.transpose(2,1)
        x = x + self.mlp(self.ln2(x))
        x = x.transpose(2,1)
        x = x.mul(mask)
        return x


class GPT(nn.Module):
    """  the full GPT language model, with a context size of block_size """
    def __init__(self, n_output=33, vocab_size=20, block_size=128, n_layer=12, n_head=8, n_embd=256,
                 embd_pdrop=0.1, resid_pdrop=0.1, attn_pdrop=0.1, n_unmasked=0):
        super().__init__()
        config = GPTConfig(n_output=n_output, block_size=block_size, vocab_size=vocab_size,
                           embd_pdrop=embd_pdrop, resid_pdrop=resid_pdrop, attn_pdrop=attn_pdrop,
                           n_layer=n_layer, n_head=n_head, n_embd=n_embd, n_unmasked=n_unmasked)
        # input embedding stem
        self.drop = nn.Dropout(config.embd_pdrop)
        # transformer
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        # decoder head
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.n_output, bias=False)
        self.block_size = config.block_size
        self.apply(self._init_weights)
        self.config = config
        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, embeddings):
        # forward the GPT model
        x = self.drop(embeddings)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)

        return logits


