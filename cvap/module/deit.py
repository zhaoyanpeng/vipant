import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

import timm
from timm.models.layers import to_2tuple, trunc_normal_
from timm.models.vision_transformer import VisionTransformer

class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(
        self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True, stride=None
    ):
        super().__init__()
        if isinstance(patch_size, dict): # hack
            patch_size, stride = patch_size["patch_size"], patch_size["stride"]
        img_size = list(to_2tuple(img_size))
        patch_size = list(to_2tuple(patch_size))
        self.img_size = img_size
        self.patch_size = patch_size

        stride = stride or patch_size
        if isinstance(stride, int):
            stride = [stride] * 2
        stride = list(stride)

        row_stride, col_stride = stride[:2]
        nrow = (img_size[0] - patch_size[0]) // row_stride + 1
        ncol = (img_size[1] - patch_size[1]) // col_stride + 1

        self.grid_size = (nrow, ncol)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        if x.shape[1] != self.proj.weight.shape[1]: # interpolate weight
            conv1_weight = self.proj.weight.mean(1, keepdim=True)
            x = F.conv2d(
                x, conv1_weight, bias=self.proj.bias, stride=self.proj.stride
            )
        else:
            x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x

class DistilledVisionTransformer(VisionTransformer):
    def __init__(self, *args, output_dim=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.dist_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 2, self.embed_dim))
        self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if self.num_classes > 0 else nn.Identity()

        trunc_normal_(self.dist_token, std=.02)
        trunc_normal_(self.pos_embed, std=.02)
        self.head_dist.apply(self._init_weights)

        scale = self.embed_dim ** -0.5
        self.proj = nn.Parameter(scale * torch.randn(self.embed_dim, output_dim)) if output_dim is not None else None

    def forward_features(self, x):
        # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
        # with slight modifications to add the dist_token
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        dist_token = self.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        # non-linear operator because of Tanh activation function, it might be desired because we want to use
        # classification head as the head for contrastive learning 
        # still, we only want a simple projection layer
        if self.proj is not None:
            x = x[:, :2] @ self.proj
        else:
            x = self.pre_logits(x)
        return x[:, 0], x[:, 1]

    def forward(self, x):
        x, x_dist = self.forward_features(x)
        x = self.head(x)
        x_dist = self.head_dist(x_dist)
        if self.training:
            return x, x_dist
        else:
            # during inference, return the average of both classifier predictions
            return (x + x_dist) / 2

