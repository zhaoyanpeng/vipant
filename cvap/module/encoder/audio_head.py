from collections import OrderedDict
from typing import Tuple, Union
from fvcore.common.registry import Registry
from omegaconf.listconfig import ListConfig

import math
import copy
import threading
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .. import ModifiedResNet, VisualTransformer

AUDIO_HEADS_REGISTRY = Registry("AUDIO_HEADS")
AUDIO_HEADS_REGISTRY.__doc__ = """
Registry for audio encoders.
"""

def build_audio_head(cfg, **kwargs):
    return AUDIO_HEADS_REGISTRY.get(cfg.name)(cfg, **kwargs)

@AUDIO_HEADS_REGISTRY.register()
class AudioHead(nn.Module):
    def __init__(self, cfg, **kwargs):
        super().__init__()
        if isinstance(cfg.layers, (tuple, list, ListConfig)):
            heads = cfg.width * 32 // 64
            self.encoder = ModifiedResNet(
                in_channels=1,
                input_resolution=cfg.resolution,
                output_dim=cfg.embed_dim,
                layers=cfg.layers,
                width=cfg.width,
                heads=heads,
            )
        else:
            heads = cfg.width // 64
            self.encoder = VisualTransformer(
                in_channels=1,
                stride=cfg.stride,
                input_resolution=cfg.resolution,
                output_dim=cfg.embed_dim,
                patch_size=cfg.patch_size,
                layers=cfg.layers,
                width=cfg.width,
                heads=heads,
            )

    def copy_state_dict(self, state_dict): 
        excluded = ["conv1.weight", "positional_embedding", "attnpool.positional_embedding"]
        new_dict = self.encoder.state_dict()
        old_dict = {k: v for k, v in state_dict.items() if k not in excluded}
        # conv1: 3 channels -> 1 channel
        old_dict["conv1.weight"] = state_dict["conv1.weight"].mean(1, keepdim=True)
        # interpolate positional embedding
        if not isinstance(self.encoder, ModifiedResNet):
            pos_resolution = self.encoder.position_resolution
            old_pos_emb = state_dict["positional_embedding"]
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
            old_dict["positional_embedding"] = new_pos_emb 
        else:
            pos_resolution = self.encoder.position_resolution
            old_pos_emb = state_dict["attnpool.positional_embedding"]
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
            old_dict["attnpool.positional_embedding"] = new_pos_emb 
        new_dict.update(old_dict)
        self.encoder.load_state_dict(new_dict)

    def forward(self, audios, *args, **kwargs):
        z = self.encoder(audios)
        if kwargs.get("normalized", False):
            z = z / z.norm(dim=-1, keepdim=True)
            #print(f"{threading.current_thread().ident} audio --{kwargs.get('normalized', False)}")
        return z 
