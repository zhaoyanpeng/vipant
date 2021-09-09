from collections import OrderedDict
from typing import Tuple, Union
from fvcore.common.registry import Registry
from omegaconf.listconfig import ListConfig
from functools import partial

import re
import math
import copy
import warnings
import threading
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from timm.models.layers import to_2tuple
from .. import ModifiedResNet, VisualTransformer, DistilledVisionTransformer, PatchEmbed

AUDIO_HEADS_REGISTRY = Registry("AUDIO_HEADS")
AUDIO_HEADS_REGISTRY.__doc__ = """
Registry for audio encoders.
"""

def build_audio_head(cfg, **kwargs):
    return AUDIO_HEADS_REGISTRY.get(cfg.name)(cfg, **kwargs)

def position_resolution(input_resolution, patch_size, stride):
    input_resolution = list(to_2tuple(input_resolution))
    patch_size = list(to_2tuple(patch_size))

    stride = stride or patch_size
    if isinstance(stride, int):
        stride = [stride] * 2
    stride = list(stride)

    row_stride, col_stride = stride[:2]
    nrow = (input_resolution[0] - patch_size[0]) // row_stride + 1
    ncol = (input_resolution[1] - patch_size[1]) // col_stride + 1
    return nrow, ncol

def interp_conv_weight(old_dict, new_dict, key):
    old_conv_weight = old_dict[key]
    new_conv_weight = new_dict[key]
    if new_conv_weight.shape[2:] != old_conv_weight.shape[2:]:
        old_conv_weight = F.interpolate(
            old_conv_weight,
            new_conv_weight.shape[2:],
            mode="bilinear",
            align_corners=False,
        )
    return old_conv_weight

def interp_pos_embedding(state_dict, old_dict, new_dict, key, bop, pos_resolution):
    """bop: start position of the postional embeddings"""
    add_leading_dim = False
    old_pos_emb = state_dict[key]
    if old_pos_emb.dim() == 3: # ensure of rank-2 tensor
        assert old_pos_emb.shape[0] == 1
        old_pos_emb = old_pos_emb.squeeze(0)
        add_leading_dim = True
    num_pos, pos_dim = old_pos_emb.shape[-2:]

    num_pos = int(np.sqrt(num_pos - bop))
    ptensor = old_pos_emb[bop:].reshape(
        -1, num_pos, num_pos, pos_dim
    ).permute(0, 3, 1, 2)

    new_pos_emb = F.interpolate(
        ptensor,
        pos_resolution,
        mode="bilinear",
        align_corners=False,
    ).permute(0, 2, 3, 1).flatten(1, 2)
    new_pos_emb = torch.cat((
        old_pos_emb[:bop], new_pos_emb.view(-1, pos_dim)
    ), dim=0)
    new_pos_emb = new_pos_emb.unsqueeze(0) if add_leading_dim else new_pos_emb
    old_dict[key] = new_pos_emb

    new_keys = set(new_dict.keys())
    old_keys = set(old_dict.keys())
    new_dict.update(old_dict)
    n_o = new_keys - old_keys
    o_n = old_keys - new_keys
    #print(f"{n_o}\n{o_n}")
    return n_o, o_n

def load_pos_embedding(
    state_dict, old_dict, new_dict, key, bop, old_pos_shape, new_pos_shape, use_slice=True
):
    add_leading_dim = False
    old_pos_emb = state_dict[key]
    if old_pos_emb.dim() == 3: # ensure of rank-2 tensor
        assert old_pos_emb.shape[0] == 1
        old_pos_emb = old_pos_emb.squeeze(0)
        add_leading_dim = True
    num_pos, pos_dim = old_pos_emb.shape[-2:]
    num_pos_required = np.prod(new_pos_shape)

    if new_pos_shape == old_pos_shape:
        new_pos_emb = old_pos_emb # do nothing
    elif use_slice and new_pos_shape[-1] == old_pos_shape[-1] and num_pos_required + bop <= num_pos:
        new_pos_emb = old_pos_emb[:num_pos_required + bop] # first k time steps
    else: # interpolate
        shape = (-1,) + old_pos_shape + (pos_dim,)
        ptensor = old_pos_emb[bop:].reshape(shape).permute(0, 3, 1, 2)
        new_pos_emb = F.interpolate(
            ptensor,
            new_pos_shape,
            mode="bilinear",
            align_corners=False,
        ).permute(0, 2, 3, 1).flatten(1, 2)
        new_pos_emb = torch.cat((
            old_pos_emb[:bop], new_pos_emb.view(-1, pos_dim)
        ), dim=0)
    new_pos_emb = new_pos_emb.unsqueeze(0) if add_leading_dim else new_pos_emb
    old_dict[key] = new_pos_emb

    new_keys = set(new_dict.keys())
    old_keys = set(old_dict.keys())
    new_dict.update(old_dict)
    n_o = new_keys - old_keys
    o_n = old_keys - new_keys
    #print(f"{n_o}\n{o_n}")
    return n_o, o_n

@AUDIO_HEADS_REGISTRY.register()
class NaiveCLIPAudioHead(nn.Module):
    def __init__(self, cfg, **kwargs):
        super().__init__()
        if isinstance(cfg.layers, (tuple, list, ListConfig)):
            heads = cfg.width * 32 // 64
            self.encoder = ModifiedResNet(
                in_channels=getattr(cfg, "in_channel", 1),
                input_resolution=cfg.resolution,
                output_dim=cfg.embed_dim,
                layers=cfg.layers,
                width=cfg.width,
                heads=heads,
            )
        else:
            heads = cfg.width // 64
            self.encoder = VisualTransformer(
                in_channels=getattr(cfg, "in_channel", 1),
                stride=cfg.stride,
                input_resolution=cfg.resolution,
                output_dim=cfg.embed_dim,
                patch_size=cfg.patch_size,
                layers=cfg.layers,
                width=cfg.width,
                heads=heads,
            )

    def from_pretrained(self, state_dict, cfg, *args, **kwargs):
        if (list(state_dict.keys())[0]).startswith("encoder."):
            audio_head_sd_new = OrderedDict()
            for k, v in state_dict.items():
                k = re.sub("^encoder\.", "", k)
                audio_head_sd_new[k] = v
            state_dict = audio_head_sd_new
        excluded = ["positional_embedding", "attnpool.positional_embedding"]
        new_dict = self.encoder.state_dict()
        old_dict = {k: v for k, v in state_dict.items() if k not in excluded}
        # interpolate positional embedding
        key = ("attnpool.positional_embedding"
            if isinstance(self.encoder, ModifiedResNet) else "positional_embedding"
        )
        new_pos_shape = self.encoder.position_resolution
        old_pos_shape = position_resolution(
            cfg.model.audio.resolution, cfg.model.audio.patch_size, cfg.model.audio.stride
        ) # nrow always indicates the time dimenstion
        n_o, o_n = load_pos_embedding(
            state_dict, old_dict, new_dict, key, 1, old_pos_shape, new_pos_shape
        )
        self.encoder.load_state_dict(new_dict)
        return n_o, o_n

    def copy_state_dict(self, state_dict): 
        excluded = ["conv1.weight", "positional_embedding", "attnpool.positional_embedding"]
        new_dict = self.encoder.state_dict()
        old_dict = {k: v for k, v in state_dict.items() if k not in excluded}
        # conv1: 3 channels -> 1 channel
        conv_key = "conv1.weight"
        old_conv_weight = interp_conv_weight(state_dict, new_dict, conv_key)
        old_dict[conv_key] = (old_conv_weight.mean(1, keepdim=True)
            if new_dict[conv_key].shape[1] != old_conv_weight.shape[1] else old_conv_weight
        )
        # interpolate positional embedding
        key = ("attnpool.positional_embedding"
            if isinstance(self.encoder, ModifiedResNet) else "positional_embedding"
        )
        n_o, o_n = interp_pos_embedding(
            state_dict, old_dict, new_dict, key, 1, self.encoder.position_resolution
        )
        self.encoder.load_state_dict(new_dict)
        return n_o, o_n

    def forward(self, audios, *args, **kwargs):
        z = self.encoder(audios)
        if kwargs.get("normalized", False):
            z = z / z.norm(dim=-1, keepdim=True)
            #print(f"{threading.current_thread().ident} audio --{kwargs.get('normalized', False)}")
        return z 

@AUDIO_HEADS_REGISTRY.register()
class NaiveDeiTAudioHead(nn.Module):
    def __init__(self, cfg, **kwargs):
        super().__init__()
        heads = cfg.width // 64
        self.encoder = DistilledVisionTransformer(
            img_size=cfg.resolution,
            # hack and has to be used with the customized `PatchEmbed`
            patch_size={"patch_size": cfg.patch_size, "stride": cfg.stride},
            representation_size=False,
            output_dim=cfg.embed_dim,
            embed_dim=cfg.width,
            depth=cfg.layers,
            num_heads=heads,
            mlp_ratio=4,
            qkv_bias=True,
            in_chans=getattr(cfg, "in_channel", 1),
            num_classes=-1,
            embed_layer=PatchEmbed,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            **kwargs
        )

    def from_pretrained(self, state_dict, cfg, *args, **kwargs):
        if (list(state_dict.keys())[0]).startswith("encoder."):
            audio_head_sd_new = OrderedDict()
            for k, v in state_dict.items():
                k = re.sub("^encoder\.", "", k)
                audio_head_sd_new[k] = v
            state_dict = audio_head_sd_new
        excluded = ["pos_embed"]
        new_dict = self.encoder.state_dict()
        old_dict = {k: v for k, v in state_dict.items() if k not in excluded}
        # interpolate positional embedding
        key = "pos_embed"
        new_pos_shape = self.encoder.patch_embed.grid_size
        old_pos_shape = position_resolution(
            cfg.model.audio.resolution, cfg.model.audio.patch_size, cfg.model.audio.stride
        ) # nrow always indicates the time dimenstion
        n_o, o_n = load_pos_embedding(
            state_dict, old_dict, new_dict, key, 2, old_pos_shape, new_pos_shape
        )
        self.encoder.load_state_dict(new_dict)
        return n_o, o_n

    def copy_state_dict(self, state_dict):
        excluded = ["patch_embed.proj.weight", "pos_embed"]
        new_dict = self.encoder.state_dict()
        old_dict = {k: v for k, v in state_dict.items() if k not in excluded and k in new_dict}
        # conv1: 3 channels -> 1 channel
        conv_key = "patch_embed.proj.weight"
        old_conv_weight = interp_conv_weight(state_dict, new_dict, conv_key)
        old_dict[conv_key] = (old_conv_weight.mean(1, keepdim=True)
            if new_dict[conv_key].shape[1] != old_conv_weight.shape[1] else old_conv_weight
        )
        # interpolate positional embedding
        key = "pos_embed"
        n_o, o_n = interp_pos_embedding(
            state_dict, old_dict, new_dict, key, 2, self.encoder.patch_embed.grid_size
        )
        self.encoder.load_state_dict(new_dict)
        return n_o, o_n

    def forward(self, audios, *args, **kwargs):
        cls_z, distilled_z = self.encoder.forward_features(audios)
        z = (cls_z + distilled_z) / 2
        if kwargs.get("normalized", False):
            z = z / z.norm(dim=-1, keepdim=True)
            #print(f"{threading.current_thread().ident} image --{kwargs.get('normalized', False)}")
        return z

@AUDIO_HEADS_REGISTRY.register()
class DeiTAudioHead(nn.Module):
    def __init__(self, cfg, **kwargs):
        super().__init__()
        heads = cfg.width // 64
        self.encoder = DistilledVisionTransformer(
            img_size=cfg.resolution,
            # hack and has to be used with the customized `PatchEmbed`
            patch_size={"patch_size": cfg.patch_size, "stride": cfg.stride},
            representation_size=cfg.embed_dim,
            embed_dim=cfg.width,
            depth=cfg.layers,
            num_heads=heads,
            mlp_ratio=4,
            qkv_bias=True,
            in_chans=1,
            num_classes=-1,
            embed_layer=PatchEmbed,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            **kwargs
        )
        self.time_first = cfg.time_first

    def copy_state_dict(self, state_dict):
        excluded = ["patch_embed.proj.weight", "pos_embed"]
        new_dict = self.encoder.state_dict()
        old_dict = {k: v for k, v in state_dict.items() if k not in excluded and k in new_dict}
        # conv1: 3 channels -> 1 channel
        key = "patch_embed.proj.weight"
        old_conv_weight = state_dict[key]
        new_conv_weight = new_dict[key]
        if new_conv_weight.shape[2:] != old_conv_weight.shape[2:]:
            old_conv_weight = F.interpolate(
                old_conv_weight,
                new_conv_weight.shape[2:],
                mode="bilinear",
                align_corners=False,
            )
        old_dict[key] = old_conv_weight.mean(1, keepdim=True)
        # interpolate positional embedding
        nrow, ncol = self.encoder.patch_embed.grid_size
        old_pos_emb = state_dict["pos_embed"]
        num_pos, pos_dim = old_pos_emb.shape[1:]

        num_pos = int(np.sqrt(num_pos - 2))
        ptensor = old_pos_emb[:, 2:].reshape(
            -1, num_pos, num_pos, pos_dim
        ).permute(0, 3, 1, 2)

        time_first = self.time_first # time_first is more reasonable: conv scans inputs left-to-right and top-to-down
        if not time_first: # skip the sanity check
            self.encoder.patch_embed.img_size = self.encoder.patch_embed.img_size[::-1]
        if nrow <= num_pos: # time
            left = int(round(0.5 * (num_pos- nrow)))
            ptensor = (
                ptensor[:, :, left : left + nrow, :] if time_first else
                ptensor[:, :, :, left : left + nrow]
            )
        else:
            size = (nrow, num_pos) if time_first else (num_pos, nrow)
            ptensor = F.interpolate(
                ptensor,
                size,
                mode="bilinear",
                align_corners=False,
            )
        if ncol <= num_pos: # feature
            left = int(round(0.5 * (num_pos- ncol)))
            ptensor = (
                ptensor[:, :, :, left : left + ncol] if time_first else
                ptensor[:, :, left : left + ncol, :]
            )
        else:
            size = (nrow, ncol) if time_first else (ncol, nrow)
            ptensor = F.interpolate(
                ptensor,
                size,
                mode="bilinear",
                align_corners=False,
            )
        new_pos_emb = ptensor.permute(0, 2, 3, 1).flatten(1, 2)
        new_pos_emb = torch.cat((old_pos_emb[:, :2], new_pos_emb), dim=1)
        old_dict["pos_embed"] = new_pos_emb

        new_keys = set(new_dict.keys())
        old_keys = set(old_dict.keys())
        new_dict.update(old_dict)
        self.encoder.load_state_dict(new_dict)
        n_o = new_keys - old_keys
        o_n = old_keys - new_keys
        #print(f"{n_o}\n{o_n}")
        return n_o, o_n

    def forward(self, images, *args, **kwargs):
        if not self.time_first: # default (..., time, bins)
            images = images.transpose(-1, -2)
        cls_z, distilled_z = self.encoder.forward_features(images)
        z = (cls_z + distilled_z) / 2
        if kwargs.get("normalized", False):
            z = z / z.norm(dim=-1, keepdim=True)
            #print(f"{threading.current_thread().ident} image --{kwargs.get('normalized', False)}")
        return z

@AUDIO_HEADS_REGISTRY.register()
class CLIPVisionEncoderAsAudioHead(nn.Module):
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
        self.time_first = cfg.time_first

    def from_pretrained(self, state_dict, cfg, *args, **kwargs):
        excluded = ["conv1.weight", "positional_embedding", "attnpool.positional_embedding"]
        new_dict = self.encoder.state_dict()
        old_dict = {k: v for k, v in state_dict.items() if k not in excluded}
        # conv1: 3 channels -> 1 channel
        key = "conv1.weight"
        old_conv_weight = state_dict[key]
        new_conv_weight = new_dict[key]
        if new_conv_weight.shape[2:] != old_conv_weight.shape[2:]:
            old_conv_weight = F.interpolate(
                old_conv_weight,
                new_conv_weight.shape[2:],
                mode="bilinear",
                align_corners=False,
            )
        old_dict[key] = old_conv_weight.mean(1, keepdim=True)
        # interpolate positional embedding
        if isinstance(self.encoder, ModifiedResNet):
            key = "attnpool.positional_embedding"
        else:
            key = "positional_embedding"
        nrow, ncol = self.encoder.position_resolution
        old_pos_emb = state_dict[key]
        num_pos, pos_dim = old_pos_emb.shape[:2]

        old_nrow, old_ncol = position_resolution(
            cfg.model.audio.resolution, cfg.model.audio.patch_size, cfg.model.audio.stride
        ) # nrow always indicates the time dimenstion
        old_time_first = getattr(cfg.running, "time_first", True)
        shape = (-1, old_nrow, old_ncol, pos_dim) if old_time_first else (-1, old_ncol, old_nrow, pos_dim)
        ptensor = old_pos_emb[1:].reshape(shape).permute(0, 3, 1, 2)
        if not old_time_first: # the reference tensor should always be time-first
            ptensor = ptensor.transpose(-1, -2)
        if not self.time_first: # (self.time_fist ^ old_time_first):
            self.time_first = True
            warnings.warn(
                "`self.time_first` has been reset to `True` because we always ensure of " +
                "time-first convolution kernel when loading from a pretrained model.", stacklevel=2
            )

        assert self.time_first, f"`self.time_first` must be True."

        time_first = self.time_first
        if nrow <= old_nrow: # time
            left = int(round(0.5 * (old_nrow- nrow)))
            ptensor = (
                ptensor[:, :, left : left + nrow, :] if time_first else
                ptensor[:, :, :, left : left + nrow]
            )
        else:
            size = (nrow, old_ncol) if time_first else (old_ncol, nrow)
            ptensor = F.interpolate(
                ptensor,
                size,
                mode="bilinear",
                align_corners=False,
            )
        if ncol <= old_ncol: # feature
            left = int(round(0.5 * (old_ncol- ncol)))
            ptensor = (
                ptensor[:, :, :, left : left + ncol] if time_first else
                ptensor[:, :, left : left + ncol, :]
            )
        else:
            size = (nrow, ncol) if time_first else (ncol, nrow)
            ptensor = F.interpolate(
                ptensor,
                size,
                mode="bilinear",
                align_corners=False,
            )
        new_pos_emb = ptensor.permute(0, 2, 3, 1).flatten(1, 2)
        new_pos_emb = torch.cat((
            old_pos_emb[:1], new_pos_emb.view(-1, pos_dim)
        ), dim=0)
        old_dict[key] = new_pos_emb

        new_keys = set(new_dict.keys())
        old_keys = set(old_dict.keys())
        new_dict.update(old_dict)
        self.encoder.load_state_dict(new_dict)
        n_o = new_keys - old_keys
        o_n = old_keys - new_keys
        #print(f"{n_o}\n{o_n}")
        return n_o, o_n

    def copy_state_dict(self, state_dict):
        excluded = ["conv1.weight", "positional_embedding", "attnpool.positional_embedding"]
        new_dict = self.encoder.state_dict()
        old_dict = {k: v for k, v in state_dict.items() if k not in excluded}
        # conv1: 3 channels -> 1 channel
        key = "conv1.weight"
        old_conv_weight = state_dict[key]
        new_conv_weight = new_dict[key]
        if new_conv_weight.shape[2:] != old_conv_weight.shape[2:]:
            old_conv_weight = F.interpolate(
                old_conv_weight,
                new_conv_weight.shape[2:],
                mode="bilinear",
                align_corners=False,
            )
        old_dict[key] = old_conv_weight.mean(1, keepdim=True)
        # interpolate positional embedding
        if isinstance(self.encoder, ModifiedResNet):
            key = "attnpool.positional_embedding"
        else:
            key = "positional_embedding"
        nrow, ncol = self.encoder.position_resolution
        old_pos_emb = state_dict[key]
        num_pos, pos_dim = old_pos_emb.shape[:2]

        num_pos = int(np.sqrt(num_pos - 1))
        ptensor = old_pos_emb[1:].reshape(
            -1, num_pos, num_pos, pos_dim
        ).permute(0, 3, 1, 2)

        time_first = self.time_first # time_first is more reasonable: conv scans inputs left-to-right and top-to-down 
        if not time_first: # skip the sanity check
            pass #self.encoder.patch_embed.img_size = self.encoder.patch_embed.img_size[::-1]
        if nrow <= num_pos: # time
            left = int(round(0.5 * (num_pos- nrow)))
            ptensor = (
                ptensor[:, :, left : left + nrow, :] if time_first else
                ptensor[:, :, :, left : left + nrow]
            )
        else:
            size = (nrow, num_pos) if time_first else (num_pos, nrow)
            ptensor = F.interpolate(
                ptensor,
                size,
                mode="bilinear",
                align_corners=False,
            )
        if ncol <= num_pos: # feature
            left = int(round(0.5 * (num_pos- ncol)))
            ptensor = (
                ptensor[:, :, :, left : left + ncol] if time_first else
                ptensor[:, :, left : left + ncol, :]
            )
        else:
            size = (nrow, ncol) if time_first else (ncol, nrow)
            ptensor = F.interpolate(
                ptensor,
                size,
                mode="bilinear",
                align_corners=False,
            )
        new_pos_emb = ptensor.permute(0, 2, 3, 1).flatten(1, 2)
        new_pos_emb = torch.cat((
            old_pos_emb[:1], new_pos_emb.view(-1, pos_dim)
        ), dim=0)
        old_dict[key] = new_pos_emb

        new_keys = set(new_dict.keys())
        old_keys = set(old_dict.keys())
        new_dict.update(old_dict)
        self.encoder.load_state_dict(new_dict)
        n_o = new_keys - old_keys
        o_n = old_keys - new_keys
        #print(f"{n_o}\n{o_n}")
        return n_o, o_n

    def forward(self, audios, *args, **kwargs):
        if not self.time_first: # default (..., time, bins)
            audios = audios.transpose(-1, -2)
        z = self.encoder(audios)
        if kwargs.get("normalized", False):
            z = z / z.norm(dim=-1, keepdim=True)
            #print(f"{threading.current_thread().ident} audio --{kwargs.get('normalized', False)}")
        return z

@AUDIO_HEADS_REGISTRY.register()
class DeiTAudioEncoderHead(DeiTAudioHead):
    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)

    def forward(self, audios, *args, **kwargs):
        if not self.time_first: # default (..., time, bins)
            audios = audios.transpose(-1, -2)
        pass

@AUDIO_HEADS_REGISTRY.register()
class CLIPAudioEncoderHead(CLIPVisionEncoderAsAudioHead):
    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)

    def forward(self, audios, *args, **kwargs):
        if not self.time_first: # default (..., time, bins)
            audios = audios.transpose(-1, -2)
        z, features = self.encoder(audios, require_feature=True)
        nrow, ncol = self.encoder.position_resolution
        N, _, D = features.shape
        if self.time_first:
            features = features.view(N, nrow, ncol, D)
        else:
            features = features.view(N, ncol, nrow, D)
        if kwargs.get("normalized", False):
            z = z / z.norm(dim=-1, keepdim=True)
            #print(f"{threading.current_thread().ident} audio --{kwargs.get('normalized', False)}")
        return z, features
