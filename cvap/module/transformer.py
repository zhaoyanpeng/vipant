from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from clip import LayerNorm, QuickGELU

class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        if self.attn_mask is not None:
            self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device)    
            attn_mask = self.attn_mask[:x.shape[0], :x.shape[0]]
        else:
            attn_mask = None 
        return self.attn(x, x, x, need_weights=False, attn_mask=attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GeneralResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, require_inter_attn = False):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

        self.require_inter_attn = require_inter_attn
        if self.require_inter_attn:
            self.attn_inter_ln = LayerNorm(d_model)
            self.attn_inter = nn.MultiheadAttention(d_model, n_head)

    def attention(self, x: torch.Tensor):
        if self.attn_mask is not None:
            self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device)
            attn_mask = self.attn_mask[:x.shape[0], :x.shape[0]]
        else:
            attn_mask = None
        return self.attn(x, x, x, need_weights=False, attn_mask=attn_mask)[0]

    def forward(self, x):
        if isinstance(x, tuple):
            x, memory = x
        else:
            memory = None
        x = x + self.attention(self.ln_1(x))
        if self.require_inter_attn:
            x = self.attn_inter_ln(x)
            x = x + self.attn_inter(x, memory, memory, need_weights=False)[0]
        x = x + self.mlp(self.ln_2(x))
        return x, memory

class Transformer(nn.Module):
    def __init__(
        self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, require_inter_attn: bool = False
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[
            #ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)
            GeneralResidualAttentionBlock(width, heads, attn_mask, require_inter_attn) for _ in range(layers)
        ])

    def forward(self, x: torch.Tensor, memory: torch.Tensor=None):
        #return self.resblocks(x)
        return self.resblocks((x, memory))[0]
