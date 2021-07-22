from collections import OrderedDict
from typing import Tuple, Union
from fvcore.common.registry import Registry

import copy
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from clip import Transformer, ModifiedResNet, VisualTransformer  

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
        pass

