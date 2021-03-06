from collections import OrderedDict
from typing import Tuple, Union
from fvcore.common.registry import Registry

import copy
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from clip import LayerNorm
from . import Transformer

class TextualTransformer(nn.Module):
    def __init__(
            self, 
            width: int, 
            layers: int, 
            heads: int, 
            ctx_len: int,
            vocab_size: int,
            output_dim: int, 
            require_inter_attn: bool = False,
        ):
        super().__init__()
        self.ctx_len = ctx_len
        self.transformer = Transformer(
            width=width,
            layers=layers,
            heads=heads,
            attn_mask=self.build_attention_mask(),
            require_inter_attn=require_inter_attn,
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, width)
        self.positional_embedding = nn.Parameter(torch.empty(ctx_len, width))
        self.ln_final = LayerNorm(width)

        self.text_projection = nn.Parameter(torch.empty(width, output_dim))

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.ctx_len, self.ctx_len)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.token_embedding.weight.dtype

    def forward(self, text, positional_embedding=None, memory=None, require_feature=False):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]
        
        positional_embedding = positional_embedding or self.positional_embedding
        positional_embedding = positional_embedding[:x.shape[1]]
        x = x + positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x, memory)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = x_feature = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        if require_feature:
            return x, x_feature

        return x

