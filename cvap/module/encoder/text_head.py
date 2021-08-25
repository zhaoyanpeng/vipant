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

@TEXT_HEADS_REGISTRY.register()
class SeqGenerationHead(nn.Module):
    def __init__(self, cfg, **kwargs):
        super().__init__()
        self.encoder = TextualTransformer(
            width=cfg.width,
            layers=cfg.layers,
            heads=cfg.heads,
            ctx_len=cfg.ctx_len,
            vocab_size=cfg.vocab_size,
            output_dim=cfg.embed_dim,
            require_inter_attn=True,
        )
        width = cfg.width
        scale = width ** -0.5
        self.mem_ln = LayerNorm(width)
        self.to_txt = nn.Parameter(scale * torch.randn(cfg.mem_width, cfg.width))
        self.predictor = nn.Linear(width, self.encoder.vocab_size, bias=cfg.bias)
        self.max_len_dec = cfg.max_len_dec

    def copy_state_dict(self, state_dict):
        excluded = []
        new_dict = self.encoder.state_dict()
        old_dict = {k: v for k, v in state_dict.items() if k not in excluded}
        new_keys = set(new_dict.keys())
        old_keys = set(old_dict.keys())
        new_dict.update(old_dict)
        self.encoder.load_state_dict(new_dict)
        n_o = new_keys - old_keys
        o_n = old_keys - new_keys
        #print(f"{n_o}\n{o_n}")
        return n_o, o_n

    def infer(self, x, positional_embedding, memory):
        beg_len = 0
        max_len = self.max_len_dec - beg_len
        logits = list()
        indice = torch.arange(0, x.shape[0], 5, device=x.device)
        x = x[indice]
        if beg_len > 0: # gold prefix and fake logits
            all_ctx = x[:, :beg_len + 1]
            logit = torch.zeros((
                all_ctx.size(0), beg_len, self.encoder.vocab_size
            ), device=x.device)
            logit = logit.scatter(2, all_ctx[:, 1:].unsqueeze(-1), 10)
            logits.append(logit)
        else: # the start symbol
            all_ctx = x[:, :1]

        for istep in range(beg_len, max_len):
            _, features = self.encoder(
                all_ctx, positional_embedding=positional_embedding, memory=memory, require_feature=True
            )
            logit = self.predictor(features[:, -1:])
            logits.append(logit)

            new_ctx = logit.argmax(dim=-1)
            all_ctx = torch.cat((all_ctx, new_ctx), 1)

        logits = torch.cat(logits, dim=1)
        return x, logits, all_ctx

    def forward(self, text, audio, time_first, *args, **kwargs):
        # layer-normed audio: (N, nrow, ncol, D)
        audio = audio @ self.to_txt # project to the textual space
        audio = audio.mean(2) if time_first else audio.mean(1)
        audio = self.mem_ln(audio).permute(1, 0, 2)  # NLD -> LND
        # text conditional on audio
        positional_embedding = kwargs.get("positional_embedding", None)

        if not self.training:
            return self.infer(text, positional_embedding, audio)

        z, features = self.encoder(
            text, positional_embedding=positional_embedding, memory=audio, require_feature=True
        )
        logits = self.predictor(features) # compute cross-entropy loss
        logits = logits[:, :-1]

        if kwargs.get("normalized", False):
            z = z / z.norm(dim=-1, keepdim=True)
            #print(f"{threading.current_thread().ident} image --{kwargs.get('normalized', False)}")
        return z, logits, None
