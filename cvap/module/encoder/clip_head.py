from fvcore.common.registry import Registry
from omegaconf.listconfig import ListConfig
from collections import OrderedDict

import re
import math
import copy
import threading
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .. import build_encoder_module

""" The idea is to abstract an encoding head as a four-layer encoder. 
    (1) backbone encoder (most likely to be shared)
    (2-3) modality-specific pre- / post-encoding layer
    (4) class / positional embedding (likely to be shared) 
"""

class MetaHead(nn.Module):
    def __init__(self, cfg, **kwargs):
        super().__init__()
        kwargs.update({
            "width": cfg.width, "embed_dim": cfg.embed_dim, 
            "ctx_len": cfg.ctx_len, "resolution": cfg.resolution
        }) # shared hyperparameters

        self.encoder = build_encoder_module(
            cfg.encoder, **kwargs
        ) # backbone
        self.pre_encoder = build_encoder_module(
            cfg.pre_encoder, **kwargs
        )
        self.pre_encoder_addon = build_encoder_module(
            cfg.pre_encoder_addon, **kwargs
        ) # in-between `pre_encoder` & `encoder`
        self.post_encoder = build_encoder_module(
            cfg.post_encoder, **kwargs
        )
        self.post_encoder_addon = build_encoder_module(
            cfg.post_encoder_addon, **kwargs
        ) # in-between `encoder` & `post_encoder`

        position_resolution = (
            self.pre_encoder.position_resolution or \
            self.encoder.position_resolution or \
            self.post_encoder.position_resolution
        )
        kwargs.update({
            "position_resolution": position_resolution
        })
        self.misc = build_encoder_module(cfg.misc, **kwargs)

    def forward(self, x: torch.Tensor, *args, **kwargs):
        kwargs.update({
            "positional_embedding": self.misc.positional_embedding, 
            "class_embedding": self.misc.class_embedding
        })
        x = self.pre_encoder(x, **kwargs) # (N, L, D)
        x = self.pre_encoder_addon(x, **kwargs) # (N, L, D)
        
        # TODO assumed 3d `x`
        x = x.permute(1, 0, 2) if not self.encoder.batch_first else x # (N, L, D) -> (L, N, D)
        x = self.encoder(x, **kwargs) 
        x = x.permute(1, 0, 2) if not self.encoder.batch_first else x # (L, N, D) -> (N, L, D)

        x = self.post_encoder_addon(x, **kwargs) 
        x = self.post_encoder(x, **kwargs) 
        return x

class CLIPImageHead(MetaHead):
    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)
    
    def copy_state_dict(self, state_dict):
        if not self.encoder.batch_first: # TransformerBackbone
            pre_keys = {"conv1.weight"}
            post_keys = {"proj"}
            misc_keys = {"positional_embedding", "class_embedding"}
            old_dict = OrderedDict()
            for k, v in state_dict.items():
                if k in pre_keys:
                    k = f"pre_encoder.{k}" 
                elif k in post_keys:
                    k = f"post_encoder.{k}"
                elif k in misc_keys:
                    k = f"misc.{k}"
                else:
                    #k = re.sub("^ln_\w+\.", "ln.", k)
                    k = re.sub("^transformer\.", "encoder.", k)
                    k = re.sub("^ln_pre\.", "pre_encoder.ln.", k)
                    k = re.sub("^ln_post\.", "post_encoder.ln.", k)
                old_dict[k] = v
        else: # ResNetBackbone
            old_dict = OrderedDict()
            for k, v in state_dict.items():
                if re.match("layer\d+\.", k):
                    k = f"encoder.{k}"
                elif re.match("attnpool\.", k):
                    k = re.sub("^attnpool\.", "post_encoder.", k)
                else:
                    k = f"pre_encoder.{k}"
                old_dict[k] = v
            pos_key = "post_encoder.positional_embedding" 
            new_key = "misc." + pos_key.rsplit(".")[-1]
            old_dict[new_key] = old_dict.pop(pos_key)
        new_dict = self.state_dict()
        new_keys = set(new_dict.keys())
        old_keys = set(old_dict.keys())
        new_dict.update(old_dict)
        self.load_state_dict(new_dict)
        n_o = new_keys - old_keys
        o_n = old_keys - new_keys
        #print(f"{n_o}\n{o_n}")
        return n_o, o_n

class CLIPAudioHead(MetaHead):
    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)

    def _positional_embedding(self, old_pos_emb):
        pos_resolution = self.misc.position_resolution
        num_pos, pos_dim = old_pos_emb.shape[:2]
        num_pos_required = np.prod(pos_resolution)
        if num_pos_required + 1 <= num_pos:
            new_pos_emb = old_pos_emb[:num_pos_required + 1]
        else:
            num_pos = int(np.sqrt(num_pos - 1))
            ptensor = old_pos_emb[1:].reshape(
                -1, num_pos, num_pos, pos_dim
            ).permute(0, 3, 1, 2) 
            new_pos_emb = F.interpolate(
                ptensor,
                pos_resolution,
                mode="bilinear",
                align_corners=False,
            ).permute(0, 2, 3, 1).flatten(1, 2) 
            new_pos_emb = torch.cat((
                old_pos_emb[:1], new_pos_emb.view(-1, pos_dim)
            ), dim=0)
        return new_pos_emb

    def copy_state_dict(self, state_dict):
        if not self.encoder.batch_first: # TransformerBackbone
            pre_keys = {"conv1.weight"}
            post_keys = {"proj"}
            misc_keys = {"positional_embedding", "class_embedding"}
            old_dict = OrderedDict()
            for k, v in state_dict.items():
                if k in pre_keys:
                    k = f"pre_encoder.{k}" 
                elif k in post_keys:
                    k = f"post_encoder.{k}"
                elif k in misc_keys:
                    k = f"misc.{k}"
                else:
                    #k = re.sub("^ln_\w+\.", "ln.", k)
                    k = re.sub("^transformer\.", "encoder.", k)
                    k = re.sub("^ln_pre\.", "pre_encoder.ln.", k)
                    k = re.sub("^ln_post\.", "post_encoder.ln.", k)
                old_dict[k] = v
            # interpolation
            pos_key = "misc.positional_embedding"
            old_dict[pos_key] = self._positional_embedding(
                old_dict.pop(pos_key)
            )
        else: # ResNetBackbone
            old_dict = OrderedDict()
            for k, v in state_dict.items():
                if re.match("layer\d+\.", k):
                    k = f"encoder.{k}"
                elif re.match("attnpool\.", k):
                    k = re.sub("^attnpool\.", "post_encoder.", k)
                else:
                    k = f"pre_encoder.{k}"
                old_dict[k] = v
            # interpolation
            pos_key = "post_encoder.positional_embedding" 
            new_key = "misc." + pos_key.rsplit(".")[-1]
            old_dict[new_key] = self._positional_embedding(
                old_dict.pop(pos_key)
            )
        new_dict = self.state_dict()
        new_keys = set(new_dict.keys())
        old_keys = set(old_dict.keys())
        new_dict.update(old_dict)
        self.load_state_dict(new_dict)
        n_o = new_keys - old_keys
        o_n = old_keys - new_keys
        #print(f"{n_o}\n{o_n}")
        return n_o, o_n

class CLIPTextHead(MetaHead):
    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)
        self.initialize_parameters()

    def initialize_parameters(self):
        pass #nn.init.normal_(self.positional_embedding, std=0.01)
    
    def copy_state_dict(self, state_dict):
        pre_keys = {"token_embedding.weight"}
        post_keys = {}
        misc_keys = {"positional_embedding"}
        old_dict = OrderedDict()
        for k, v in state_dict.items():
            if k in pre_keys:
                k = f"pre_encoder.{k}" 
            elif k in post_keys:
                k = f"post_encoder.{k}"
            elif k in misc_keys:
                k = f"misc.{k}"
            else:
                #k = re.sub("^ln_\w+\.", "ln.", k)
                k = re.sub("^transformer\.", "encoder.", k)
                k = re.sub("^ln_final\.", "post_encoder.ln.", k)
                k = re.sub("^text_projection", "post_encoder.proj", k)
            old_dict[k] = v
        new_dict = self.state_dict()
        # TODO better via interpolation
        pos_key = "misc.positional_embedding"
        old_num = old_dict[pos_key].shape[0]
        new_num = new_dict[pos_key].shape[0]
        if old_num >= new_num:
            old_dict[pos_key] = old_dict.pop(pos_key)[:new_num]
        else:
            new_dict[pos_key][:old_num] = old_dict.pop(pos_key)
            old_dict[pos_key] = new_dict[pos_key] # unnecessary
        new_keys = set(new_dict.keys())
        old_keys = set(old_dict.keys())
        new_dict.update(old_dict)
        self.load_state_dict(new_dict)
        n_o = new_keys - old_keys
        o_n = old_keys - new_keys
        #print(f"{n_o}\n{o_n}")
        return n_o, o_n
