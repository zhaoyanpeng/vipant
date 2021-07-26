from collections import OrderedDict
from typing import Tuple, Union
from fvcore.common.registry import Registry

import copy
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from clip import LayerNorm
from .. import Transformer

TEXT_HEADS_REGISTRY = Registry("TEXT_HEADS")
TEXT_HEADS_REGISTRY.__doc__ = """
Registry for text encoders.
"""

def build_text_head(cfg, **kwargs):
    return TEXT_HEADS_REGISTRY.get(cfg.name)(cfg, **kwargs)

@TEXT_HEADS_REGISTRY.register()
class TextHead(nn.Module):
    def __init__(self, cfg, **kwargs):
        super().__init__()
        self.ctx_len = cfg.ctx_len
        self.encoder = Transformer(
            width=cfg.width,
            layers=cfg.layers,
            heads=cfg.heads,
            attn_mask=self.build_attention_mask()
        )

        self.vocab_size = cfg.vocab_size
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.width)
        self.positional_embedding = nn.Parameter(torch.empty(self.ctx_len, cfg.width))
        self.ln_final = LayerNorm(cfg.width)

        self.text_projection = nn.Parameter(torch.empty(cfg.width, cfg.embed_dim))

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        proj_std = (self.encoder.width ** -0.5) * ((2 * self.encoder.layers) ** -0.5)
        attn_std = self.encoder.width ** -0.5
        fc_std = (2 * self.encoder.width) ** -0.5
        for block in self.encoder.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.encoder.width ** -0.5)

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

    def copy_state_dict(self, state_dict): 
        self.load_state_dict(state_dict)

    def encode(self, text, positional_embedding=None):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]
        
        positional_embedding = positional_embedding or self.positional_embedding
        positional_embedding = positional_embedding[:x.shape[1]]
        x = x + positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.encoder(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return x

    def forward(self, text, *args, **kwargs):
        positional_embedding = kwargs.get("positional_embedding", None)
        z = self.encode(text, positional_embedding=positional_embedding)
        if kwargs.get("normalized", False):
            z = z / z.norm(dim=-1, keepdim=True)
            #print(f"{threading.current_thread().ident} image --{kwargs.get('normalized', False)}")
        return z 
