from collections import OrderedDict
from typing import Tuple, Union
from fvcore.common.registry import Registry

import copy
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from clip import LayerNorm
from .. import TextualTransformer

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
        self.encoder = TextualTransformer(
            width=cfg.width,
            layers=cfg.layers,
            heads=cfg.heads,
            ctx_len=cfg.ctx_len,
            vocab_size=cfg.vocab_size,
            output_dim=cfg.embed_dim,
        )

    def copy_state_dict(self, state_dict): 
        self.encoder.load_state_dict(state_dict)

    def forward(self, text, *args, **kwargs):
        positional_embedding = kwargs.get("positional_embedding", None)
        z = self.encoder(text, positional_embedding=positional_embedding)
        if kwargs.get("normalized", False):
            z = z / z.norm(dim=-1, keepdim=True)
            #print(f"{threading.current_thread().ident} image --{kwargs.get('normalized', False)}")
        return z 
