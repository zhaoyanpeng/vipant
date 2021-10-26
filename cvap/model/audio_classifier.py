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


class AudioClassifier(nn.Module):
    def __init__(self, cfg, echo):
        super(AudioClassifier, self).__init__()
        self.cfg = cfg
        self.echo = echo

    def forward(self, audios, labels, *args, **kwargs):
        device_ids = kwargs.get("device_ids", [0])
        # how to asynchronize the two `data_parallel` 
        kwargs = {"normalized": self.loss_head.normalized, "names": kwargs.get("names", None)}
        audio_features = data_parallel(
            self.audio_head, audios, device_ids=device_ids, module_kwargs=kwargs
        )
        loss = self.loss_head(audio_features, labels, **kwargs)
        return loss     
    
    def encode_text(self, text, *args, device_ids=[0], **kwargs):
        text_features = data_parallel(
            self.text_head, text, device_ids=device_ids, module_kwargs=kwargs
        )
        return text_features

    def collect_audio_state_dict(self):
        return (
            self.audio_head.state_dict(), 
            self.loss_head.state_dict(),
        )

    def report(self, gold_file=None, **kwargs):
        if not dist.is_initialized() or dist.get_rank() == 0:
            return self.loss_head.report(gold_file=gold_file, **kwargs)
        else:
            return ""
    
    def build(self, **kwargs):
        tunable_params = dict()
        if self.cfg.eval:
            local_cfg, _, audio_head_sd, _, loss_head_sd = load_checkpoint(self.cfg, self.echo)
            from_scratch, image_head_sd, text_head_sd, _ = load_clip(None, self.cfg, self.echo)
            
            self.audio_head = build_audio_head(self.cfg.model.audio)
            if audio_head_sd is not None:
                n_o, o_n = self.audio_head.from_pretrained(audio_head_sd, local_cfg)
                msg = f" except {n_o}" if len(n_o) > 0 else ""
                self.echo(f"Initialize audio encoder from `audio_head`{msg}.")
            else:
                self.audio_head.copy_state_dict(image_head_sd)
                self.echo("Initialize audio encoder from `image_head`.")

            self.text_head = build_text_head(self.cfg.model.text) #
            n_o, o_n = self.text_head.copy_state_dict(text_head_sd)
            msg = f" except {n_o}" if len(n_o) > 0 else ""
            self.echo(f"Initialize text encoder from `text_head`{msg}.")

            self.loss_head = build_loss_head(self.cfg.model.loss, **kwargs)
            if loss_head_sd is not None:
                self.loss_head.copy_state_dict(loss_head_sd) #

            self.cuda(self.cfg.rank) 
        else:
            # try pre-trained model!
            local_cfg, _, audio_head_sd, _, loss_head_sd = load_checkpoint(self.cfg, self.echo)
            # try clip! TODO do we always have to load CLIP?
            from_scratch, image_head_sd, _, model = load_clip(local_cfg, self.cfg, self.echo)
            
            #cfg = local_cfg if local_cfg is not None else self.cfg
            self.audio_head = build_audio_head(self.cfg.model.audio)
            if not self.cfg.model.audio.from_scratch:
                if local_cfg is not None:
                    if "misc.positional_embedding" in audio_head_sd:
                        self.audio_head = build_audio_head(local_cfg.model.audio)
                        self.audio_head.load_state_dict(audio_head_sd)
                    else: # backward compatible
                        if (list(audio_head_sd.keys())[0]).startswith("encoder."):
                            audio_head_sd_new = OrderedDict()
                            for k, v in audio_head_sd.items():
                                k = re.sub("^encoder\.", "", k)
                                audio_head_sd_new[k] = v
                            audio_head_sd = audio_head_sd_new
                        self.audio_head.copy_state_dict(audio_head_sd)
                    self.echo("Initialize audio encoder from `audio_head`.")
                elif not from_scratch:
                    self.audio_head.copy_state_dict(image_head_sd)
                    self.echo("Initialize audio encoder from `image_head`.")
                else:
                    self.echo("Have to learn from scratch.")
                
            self.loss_head = build_loss_head(self.cfg.model.loss, **kwargs)
            tunable_params = {
                f"loss_head.{k}": v for k, v in self.loss_head.named_parameters()
            } 
            if not self.cfg.model.audio.freeze:
                tunable_params.update({
                    f"audio_head.{k}": v for k, v in self.audio_head.named_parameters()
                })
            else:
                self.echo("Freeze audio encoder.")
            self.cuda(self.cfg.rank)
        return tunable_params

