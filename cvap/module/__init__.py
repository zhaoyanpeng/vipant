from .transformer import Transformer # has to be the first because the following depend on it
# better API
from .val import (
    build_encoder_module, ENCODER_MODULES_REGISTRY,
    interp_clip_vp_embedding,
    interp_conv_weight_channel,
    interp_conv_weight_spatial,
)
# deprecated API
from .deit import PatchEmbed, DistilledVisionTransformer
from .vit import VisualTransformer
from .txt import TextualTransformer
from .resnet import ModifiedResNet
# optimizer
from .lars import * 
# encoder heads
from .encoder import *
from .decoder import *
# dummy heads
import torch
class DummyHead(torch.nn.Module):
    def __init__(self, cfg, **kwargs):
        super().__init__()
        pass
    def from_pretrained(self, state_dict, cfg, *args, **kwargs):
        pass
    def copy_state_dict(self, state_dict):
        return {}, {}
    def replace_modules(self, **kwargs):
        return []
    def forward(self, x, *args, **kwargs):
        return None
IMAGE_HEADS_REGISTRY.register(DummyHead)
AUDIO_HEADS_REGISTRY.register(DummyHead)
TEXT_HEADS_REGISTRY.register(DummyHead)
LOSS_HEADS_REGISTRY.register(DummyHead)
