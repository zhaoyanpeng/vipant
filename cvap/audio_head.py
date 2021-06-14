from collections import OrderedDict
from typing import Tuple, Union
from fvcore.common.registry import Registry

import copy
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from clip import Transformer, ModifiedResNet  

from .module import VisualTransformer

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
        if isinstance(cfg.layers, (tuple, list)):
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
        self.initialize_parameters()

    def initialize_parameters(self):
        if isinstance(self.encoder, ModifiedResNet):
            if self.encoder.attnpool is not None:
                std = self.encoder.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.encoder.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.encoder.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.encoder.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.encoder.attnpool.c_proj.weight, std=std)

            for resnet_block in [self.encoder.layer1, self.encoder.layer2, self.encoder.layer3, self.encoder.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)

    @property
    def dtype(self):
        return self.encoder.conv1.weight.dtype

    def copy_state_dict(self, state_dict): 
        excluded = ["conv1.weight", "positional_embedding"]
        new_dict = self.encoder.state_dict()
        old_dict = {k: v for k, v in state_dict.items() if k not in excluded}
        # conv1: 3 channels -> 1 channel
        old_dict["conv1.weight"] = state_dict["conv1.weight"].mean(1, keepdim=True)
        # interpolate positional embedding
        if not isinstance(self.encoder, ModifiedResNet):
            pos_resolution = self.encoder.position_resolution
            old_pos_emb = state_dict["positional_embedding"]
            num_pos, pos_dim = old_pos_emb.shape[:2]
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
        new_dict.update(old_dict)
        self.encoder.load_state_dict(new_dict)

    def forward(self, audios, *args, **kwargs):
        return self.encoder(audios.type(self.dtype))

