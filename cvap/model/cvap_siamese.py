from omegaconf import OmegaConf
import os, re
import warnings
from typing import Union, List
from collections import defaultdict, OrderedDict

import copy
import time
import torch
import numpy as np
from torch import nn
from PIL import Image
from tqdm import tqdm

import torch.distributed as dist
from torch.nn.parallel import data_parallel
from torch.nn.parallel import DistributedDataParallel

from clip import load 

from .cvalp_dp import CVALPDP
from ..module import (
    build_image_head, build_audio_head, build_text_head, build_loss_head
)
from . import (
    load_checkpoint, load_clip, load_meme
)

class CVASPDP(CVALPDP):
    def __init__(self, cfg, echo):
        super(CVASPDP, self).__init__(cfg, echo)

    def forward(
        self, images, images_v1, audios_v1, 
        text_v1=None, images_v2=None, audios_v2=None, *args, **kwargs
    ):
        device_ids = kwargs.get("device_ids", [0])
        # how to asynchronize the two `data_parallel` 
        kwargs = {"normalized": self.loss_head.normalized, "names": kwargs.get("names", None)}
        image_features = image_features_v1 = image_features_v2 = audio_features_v1 = text_features = None
        if images is not None: # pre-computed unnormalized features
            dummy_image = list(images.shape[1:]) == [1, 1, 1]
            if self.loss_head.normalized and not dummy_image:
                images = images / images.norm(dim=-1, keepdim=True)
            image_features = images # dummy images will be ignored
        if images_v1 is not None and self.image_head is not None:
            image_features_v1 = data_parallel(
                self.image_head, images_v1, device_ids=device_ids, module_kwargs=kwargs
            )
        if images_v2 is not None and self.image_head is not None:
            image_features_v2 = data_parallel(
                self.image_head, images_v2, device_ids=device_ids, module_kwargs=kwargs
            )
        if audios_v1 is not None and self.audio_head is not None:
            audio_features_v1 = data_parallel(
                self.audio_head, audios_v1, device_ids=device_ids, module_kwargs=kwargs
            )
        if text_v1 is not None and self.text_head is not None:
            text_features = data_parallel(
                self.text_head, text_v1, device_ids=device_ids, module_kwargs=kwargs
            )
        loss = self.loss_head(
            image_features, image_features_v1, audio_features_v1,
            images_v2=image_features_v2, audios_v2=image_features_v2, **kwargs
        )
        return loss     

    def _build_siamese_backbone(self, **kwargs):
        # try pre-trained model!
        local_cfg, _, audio_head_sd, _, loss_head_sd = load_checkpoint(self.cfg, self.echo)
        from_scratch, image_head_sd, text_head_sd, _ = load_clip(None, self.cfg, self.echo)

        # image_head's parameters as the reference
        self.image_head = build_image_head(self.cfg.model.image)
        if not from_scratch and not self.cfg.model.image.from_scratch:
            n_o, o_n = self.image_head.copy_state_dict(image_head_sd)
            msg = f" except {n_o}" if len(n_o) > 0 else ""
            self.echo(f"Initialize image encoder from `image_head`{msg}.")
        if False and (self.cfg.running.frame_emb is not None or not self.cfg.running.imagine):
            self.image_head = None
            self.echo("Destory image encoder.")
        scfg = self.cfg.running.siamese

        # shared modules with audio_head
        amodules = set(scfg.amodules)
        kwargs = {
            "shared_modules": amodules, "reference": self.image_head, "keep_hp": scfg.keep_hp
        }
        self.audio_head = build_audio_head(self.cfg.model.audio, **kwargs)
        if not self.cfg.model.audio.from_scratch:
            if local_cfg is not None:
                n_o, o_n = self.audio_head.from_pretrained(audio_head_sd, local_cfg)
                msg = f" except {n_o}" if len(n_o) > 0 else ""
                self.echo(f"Initialize audio encoder from `audio_head`{msg}.")
            elif not from_scratch:
                n_o, o_n = self.audio_head.copy_state_dict(image_head_sd)
                msg = f" except {n_o}" if len(n_o) > 0 else ""
                self.echo(f"Initialize audio encoder from `image_head`{msg}.")
            else:
                self.echo("Have to learn from scratch.")
        ref_modules = self.audio_head.replace_modules(**kwargs)
        self.echo(f"A: audio_head.modules referring to image_head.modules: {ref_modules}.")

        # shared modules with text_head 
        lmodules = set(scfg.lmodules)
        kwargs.update({"shared_modules": lmodules})
        self.text_head = build_text_head(self.cfg.model.text, **kwargs)
        if not from_scratch and not self.cfg.model.text.from_scratch:
            if self.cfg.model.text.from_text:
                n_o, o_n = self.text_head.copy_state_dict(text_head_sd)
                msg = f" except {n_o}" if len(n_o) > 0 else ""
                self.echo(f"Initialize text encoder from `text_head`{msg}.")
            else:
                n_o, o_n = self.text_head.copy_state_dict(image_head_sd)
                msg = f" except {n_o}" if len(n_o) > 0 else ""
                self.echo(f"Initialize text encoder from `image_head`{msg}.")
        ref_modules = self.text_head.replace_modules(**kwargs)
        self.echo(f"T:  text_head.modules referring to image_head.modules: {ref_modules}.")
        if len(self.text_head.state_dict()) == 0:
            self.text_head = None
            self.echo("Destory text encoder.")

        self.loss_head = build_loss_head(self.cfg.model.loss)

        tunable_params = {
            f"loss_head.{k}": v for k, v in self.loss_head.named_parameters()
        } 
        if not self.cfg.model.image.freeze and self.image_head is not None:
            tunable_params.update({
                f"image_head.{k}": v for k, v in self.image_head.named_parameters()
            })
        elif self.image_head is not None:
            shared_modules = amodules | lmodules
            pattern = "|".join([f"^{m}\." for m in shared_modules])
            tunable_params.update({
                f"image_head.{k}": v for k, v in self.image_head.named_parameters()
            if pattern != "" and re.match(pattern, k)}) # shared parameters must be tunable
            self.echo(f"Freeze image encoder (excl. shared modules: {shared_modules}).")
        if not self.cfg.model.audio.freeze:
            pattern = "|".join([f"^{m}\." for m in amodules])
            tunable_params.update({
                f"audio_head.{k}": v for k, v in self.audio_head.named_parameters()
            if pattern == "" or not re.match(pattern, k)}) # filter out shared parameters
        else:
            self.echo("Freeze audio encoder.")
        if not self.cfg.model.text.freeze and self.text_head is not None:
            pattern = "|".join([f"^{m}\." for m in lmodules])
            tunable_params.update({
                f"text_head.{k}": v for k, v in self.text_head.named_parameters()
            if pattern == "" or not re.match(pattern, k)}) # filter out shared parameters
        elif self.text_head is not None:
            self.echo("Freeze text encoder.")
        return tunable_params
