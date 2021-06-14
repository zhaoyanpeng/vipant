from collections import OrderedDict
from typing import Tuple, Union
from fvcore.common.registry import Registry

import copy
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from clip import Transformer, ModifiedResNet, VisualTransformer  

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
        if isinstance(cfg.layers, (tuple, list)):
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
        self.encoder.load_state_dict(state_dict) 

    def forward(self, images, *args, **kwargs):
        return self.encoder(images.type(self.dtype))
