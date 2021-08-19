from collections import OrderedDict
from typing import Tuple, Union
from fvcore.common.registry import Registry
from omegaconf.listconfig import ListConfig

import copy
import threading
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from timm.models.vision_transformer import _cfg

from .. import ModifiedResNet, VisualTransformer, DistilledVisionTransformer

IMAGE_HEADS_REGISTRY = Registry("IMAGE_HEADS")
IMAGE_HEADS_REGISTRY.__doc__ = """
Registry for image encoders.
"""

def build_image_head(cfg, **kwargs):
    return IMAGE_HEADS_REGISTRY.get(cfg.name)(cfg, **kwargs)

@IMAGE_HEADS_REGISTRY.register()
class ImageHead(nn.Module):
    def __init__(self, cfg, **kwargs):
        super().__init__()
        if isinstance(cfg.layers, (tuple, list, ListConfig)):
            heads = cfg.width * 32 // 64
            self.encoder = ModifiedResNet(
                input_resolution=cfg.resolution,
                output_dim=cfg.embed_dim,
                layers=cfg.layers,
                width=cfg.width,
                heads=heads,
            )
        else:
            heads = cfg.width // 64
            self.encoder = VisualTransformer(
                input_resolution=cfg.resolution,
                output_dim=cfg.embed_dim,
                patch_size=cfg.patch_size,
                layers=cfg.layers,
                width=cfg.width,
                heads=heads,
            )

    def copy_state_dict(self, state_dict): 
        self.encoder.load_state_dict(state_dict) 

    def forward(self, images, *args, **kwargs):
        z = self.encoder(images)
        if kwargs.get("normalized", False):
            z = z / z.norm(dim=-1, keepdim=True)
            #print(f"{threading.current_thread().ident} image --{kwargs.get('normalized', False)}")
        return z 

@IMAGE_HEADS_REGISTRY.register()
class DeiTImageHead(nn.Module):
    def __init__(self, cfg, **kwargs):
        super().__init__()
        heads = cfg.width // 64
        self.encoder = DistilledVisionTransformer(
            img_size=cfg.resolution,
            patch_size=cfg.patch_size,
            representation_size=cfg.embed_dim,
            embed_dim=cfg.width,
            depth=cfg.layers,
            num_heads=heads,
            mlp_ratio=4,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            **kwargs
        )

    def copy_state_dict(self, state_dict):
        self.encoder.load_state_dict(state_dict)

    def forward(self, images, *args, **kwargs):
        cls_z, distilled_z = self.encoder.forward_features(images)
        z = (cls_z + distilled_z) / 2
        if kwargs.get("normalized", False):
            z = z / z.norm(dim=-1, keepdim=True)
            #print(f"{threading.current_thread().ident} image --{kwargs.get('normalized', False)}")
        return z
