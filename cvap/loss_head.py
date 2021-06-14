from collections import OrderedDict
from typing import Tuple, Union
from fvcore.common.registry import Registry

import copy
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from clip import Transformer, ModifiedResNet, VisualTransformer  

LOSS_HEADS_REGISTRY = Registry("LOSS_HEADS")
LOSS_HEADS_REGISTRY.__doc__ = """
Registry for image encoders.
"""

def build_loss_head(cfg, **kwargs):
    return LOSS_HEADS_REGISTRY.get(cfg.name)(cfg, **kwargs)

class LossHead(nn.Module):
    def __init__(self):
        super().__init__()

    def copy_state_dict(self, state_dict): 
        pass

@LOSS_HEADS_REGISTRY.register()
class CELossHead(LossHead):
    def __init__(self, cfg, **kwargs):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.loss_fct = nn.CrossEntropyLoss()
    
    def copy_state_dict(self, state_dict): 
        key = "logit_scale"
        new_dict = self.state_dict()
        new_dict.update({key: state_dict[key]})
        self.load_state_dict(new_dict)

    def forward(self, x1, x2, *args, **kwargs):
        # normalized features
        x1 = x1 / x1.norm(dim=-1, keepdim=True)
        x2 = x2 / x2.norm(dim=-1, keepdim=True)
        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_x1 = logit_scale * x1 @ x2.t()
        logits_per_x2 = logit_scale * x2 @ x1.t()
        # cross entropy loss 
        labels = torch.arange(x1.shape[0], device=x1.device)
        loss_mean_x1 = self.loss_fct(logits_per_x1, labels)
        loss_mean_x2 = self.loss_fct(logits_per_x2, labels)
        loss = loss_mean_x1 + loss_mean_x2
        return loss

@LOSS_HEADS_REGISTRY.register()
class BarlowLossHead(LossHead):
    # see Barlow Twins: https://arxiv.org/abs/2103.03230 
    def __init__(self, cfg, **kwargs):
        super().__init__()
        self.bn = nn.BatchNorm1d(cfg.embed_dim, affine=False) 
        self.off_weight = cfg.off_weight
    
    @staticmethod
    def loss_fct(c):
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = c.masked_select( # more memory-efficient?
            ~torch.eye(c.size(-1), device=c.device, dtype=torch.bool)
        ).pow_(2).sum()
        return on_diag, off_diag
    
    def forward(self, x1, x2, *args, **kwargs):
        x1, x2 = self.bn(x1), self.bn(x2)
        c = x1.t() @ x2
        c.div_(x1.size(0))
        on_diag, off_diag = self.loss_fct(c)
        loss = on_diag + self.off_weight * off_diag
        return loss

