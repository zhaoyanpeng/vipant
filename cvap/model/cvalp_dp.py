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

from ..module import (
    build_image_head, build_audio_head, build_text_head, build_loss_head
)
from . import (
    load_checkpoint, load_clip, load_meme
)

class CVALPDP(nn.Module):
    def __init__(self, cfg, echo):
        super(CVALPDP, self).__init__()
        self.cfg = cfg
        self.echo = echo

    def forward(self, images, audios, text, *args, **kwargs):
        device_ids = kwargs.get("device_ids", [0])
        # how to asynchronize the two `data_parallel` 
        kwargs = {"normalized": self.loss_head.normalized, "names": kwargs.get("names", None)}
        image_features = audio_features = text_features = None
        dummy_image = list(images.shape[1:]) == [1, 1, 1]
        if images is not None and self.image_head is not None and not dummy_image:
            image_features = data_parallel(
                self.image_head, images, device_ids=device_ids, module_kwargs=kwargs
            )
        elif images is not None: # pre-computed unnormalized features
            if self.loss_head.normalized and not dummy_image:
                images = images / images.norm(dim=-1, keepdim=True)
            image_features = images # dummy images will be ignored
        if audios is not None and self.audio_head is not None:
            audio_features = data_parallel(
                self.audio_head, audios, device_ids=device_ids, module_kwargs=kwargs
            )
        dummy_text = list(text.shape[1:]) == [1] if text is not None else True
        if text is not None and self.text_head is not None and not dummy_text:
            text_features = data_parallel(
                self.text_head, text, device_ids=device_ids, module_kwargs=kwargs
            )
        elif text is not None: # pre-computed unnormalized features
            if self.loss_head.normalized and not dummy_text:
                text = text / text.norm(dim=-1, keepdim=True)
            text_features = text # dummy text will be ignored
        loss = self.loss_head(image_features, audio_features, text_features, **kwargs)
        return loss     

    def encode_image(self, image, *args, device_ids=[0], **kwargs):
        image_features = data_parallel(
            self.image_head, images, device_ids=device_ids, module_kwargs=kwargs
        )
        return image_features
    
    def encode_audio(self, audio, *args, device_ids=[0], **kwargs):
        audio_features = data_parallel(
            self.audio_head, audios, device_ids=device_ids, module_kwargs=kwargs
        )
        return audio_features

    def encode_text(self, text, *args, device_ids=[0], **kwargs):
        text_features = data_parallel(
            self.text_head, text, device_ids=device_ids, module_kwargs=kwargs
        )
        return text_features

    def collect_audio_state_dict(self):
        return self.collect_state_dict()

    def collect_state_dict(self):
        return (
            (self.image_head.state_dict()
            if self.image_head is not None and not self.cfg.model.image.freeze else OrderedDict()),
            self.audio_head.state_dict(),
            (self.text_head.state_dict()
            if self.text_head is not None and not self.cfg.model.text.freeze else OrderedDict()),
            self.loss_head.state_dict(),
        )

    def report(self, gold_file=None, **kwargs):
        if self.training:
            return self.loss_head.stats(**kwargs) if hasattr(self.loss_head, "stats") else ""
        if not dist.is_initialized() or dist.get_rank() == 0:
            return self.loss_head.report(gold_file=gold_file, **kwargs)
        else:
            return ""
    
    def build(self, **kwargs):
        tunable_params = dict()
        if self.cfg.eval:
            local_cfg, _, audio_head_sd, _, loss_head_sd = load_checkpoint(self.cfg, self.echo)
            from_scratch, image_head_sd, text_head_sd, _ = load_clip(None, self.cfg, self.echo)
            
            self.image_head = build_image_head(self.cfg.model.image)
            self.image_head.copy_state_dict(image_head_sd)

            self.audio_head = build_audio_head(self.cfg.model.audio)
            self.audio_head.load_state_dict(audio_head_sd)

            self.text_head = build_text_head(self.cfg.model.text)
            self.text_head.copy_state_dict(text_head_sd)

            self.loss_head = build_loss_head(self.cfg.model.loss, **kwargs)
            self.loss_head.load_state_dict(loss_head_sd)

            self.cuda(self.cfg.rank) 
        else:
            if self.cfg.running.siamese.alive:
                tunable_params = self._build_siamese_backbone(**kwargs)            
            else:
                tunable_params = self._build_separate_backbone(**kwargs)            
            self.cuda(self.cfg.rank)
        return tunable_params

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
        if self.cfg.running.frame_emb is not None or not self.cfg.running.imagine:
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
        if self.cfg.running.text_emb is not None or len(self.text_head.state_dict()) == 0:
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

    def _build_separate_backbone(self, **kwargs):
        from_scratch, image_head_sd, text_head_sd, _ = load_clip(None, self.cfg, self.echo)
            
        self.image_head = build_image_head(self.cfg.model.image)
        if not from_scratch and not self.cfg.model.image.from_scratch:
            n_o, o_n = self.image_head.copy_state_dict(image_head_sd)
            msg = f" except {n_o}" if len(n_o) > 0 else ""
            self.echo(f"Initialize image encoder from `image_head`{msg}.")
        if self.cfg.running.frame_emb is not None:
            self.image_head = None
            self.echo("Destory image encoder.")

        self.audio_head = build_audio_head(self.cfg.model.audio)
        if not from_scratch and not self.cfg.model.audio.from_scratch:
            n_o, o_n = self.audio_head.copy_state_dict(image_head_sd)
            msg = f" except {n_o}" if len(n_o) > 0 else ""
            self.echo(f"Initialize audio encoder from `image_head`{msg}.")

        self.text_head = build_text_head(self.cfg.model.text)
        if not from_scratch and not self.cfg.model.text.from_scratch:
            n_o, o_n = self.text_head.copy_state_dict(text_head_sd)
            msg = f" except {n_o}" if len(n_o) > 0 else ""
            self.echo(f"Initialize text encoder from `text_head`{msg}.")
        if len(self.text_head.state_dict()) == 0:
            self.text_head = None
            self.echo("Destory text encoder.")

        self.loss_head = build_loss_head(self.cfg.model.loss, **kwargs)

        tunable_params = {
            f"loss_head.{k}": v for k, v in self.loss_head.named_parameters()
        } 
        if not self.cfg.model.image.freeze and self.image_head is not None:
            tunable_params.update({
                f"image_head.{k}": v for k, v in self.image_head.named_parameters()
            })
        elif self.image_head is not None:
            self.echo("Freeze image encoder.")
        if not self.cfg.model.audio.freeze:
            tunable_params.update({
                f"audio_head.{k}": v for k, v in self.audio_head.named_parameters()
            })
        else:
            self.echo("Freeze audio encoder.")
        if not self.cfg.model.text.freeze and self.text_head is not None:
            tunable_params.update({
                f"text_head.{k}": v for k, v in self.text_head.named_parameters()
            })
        elif self.text_head is not None:
            self.echo("Freeze text encoder.")
        return tunable_params

