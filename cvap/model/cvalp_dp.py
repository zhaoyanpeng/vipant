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

class CVALPDP(nn.Module):
    def __init__(self, cfg, echo):
        super(CVALPDP, self).__init__()
        self.cfg = cfg
        self.echo = echo

    def forward(self, images, audios, text, *args, **kwargs):
        device_ids = kwargs.get("device_ids", [0])
        # how to asynchronize the two `data_parallel` 
        kwargs = {"normalized": False, "names": kwargs.get("names", None)}
        image_features = audio_features = text_features = None
        if images is not None:
            image_features = data_parallel(
                self.image_head, images, device_ids=device_ids, module_kwargs=kwargs
            )
        if audios is not None:
            audio_features = data_parallel(
                self.audio_head, audios, device_ids=device_ids, module_kwargs=kwargs
            )
        if text is not None:
            text_features = data_parallel(
                self.text_head, text, device_ids=device_ids, module_kwargs=kwargs
            )
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
            self.image_head.state_dict(), 
            self.audio_head.state_dict(), 
            self.text_head.state_dict(), 
            self.loss_head.state_dict(),
        )

    def report(self, gold_file=None, **kwargs):
        if not dist.is_initialized() or dist.get_rank() == 0:
            return self.loss_head.report(gold_file=gold_file, **kwargs)
        else:
            return ""
    
    def build(self, **kwargs):
        tunable_params = dict()
        def load_checkpoint():
            model_file = f"{self.cfg.model_root}/{self.cfg.model_name}/{self.cfg.model_file}"
            self.echo(f"Loading from {model_file}")
            if not os.path.isfile(model_file):
                return None, None, None 
            checkpoint = torch.load(model_file, map_location="cpu")
            local_cfg = checkpoint["cfg"]
            local_str = OmegaConf.to_yaml(local_cfg)
            #self.echo(f"Old configs:\n\n{local_str}")
            image_head_sd, audio_head_sd, text_head_sd, loss_head_sd = checkpoint["model"]
            return local_cfg, image_head_sd, audio_head_sd, text_head_sd, loss_head_sd 

        if self.cfg.eval:
            local_cfg, image_head_sd, audio_head_sd, text_head_sd, loss_head_sd = load_checkpoint()
            
            self.image_head = build_image_head(local_cfg.model.image)
            self.image_head.load_state_dict(image_head_sd)

            self.audio_head = build_audio_head(local_cfg.model.audio)
            self.audio_head.load_state_dict(audio_head_sd)

            self.text_head = build_text_head(local_cfg.model.text) 
            self.text_head.load_state_dict(text_head_sd)

            self.loss_head = build_loss_head(local_cfg.model.loss, **kwargs)
            self.loss_head.load_state_dict(loss_head_sd)

            self.cuda(self.cfg.rank) 
        else:
            if self.cfg.running.siamese:
                tunable_params = self._build_siamese_backbone(**kwargs)            
            else:
                tunable_params = self._build_separate_backbone(**kwargs)            
            self.cuda(self.cfg.rank)
        return tunable_params

    def _load_clip(self, local_cfg):
        try: # try image / text backbone
            rcfg = self.cfg.running
            model, self.T = load(
                rcfg.clip_model_name, rcfg.clip_model_root, device="cpu", jit=False
            )
            image_head_sd = model.visual.state_dict() if local_cfg is None else None
            text_head_sd = OrderedDict()
            for k, v in model.state_dict().items():
                if k.startswith("visual") or k == "logit_scale":
                    continue
                #k = re.sub("^transformer\.", "encoder.", k)
                text_head_sd[k] = v
            from_scratch = False
        except Exception as e:
            self.echo(f"Will learn from scratch because: {e}") 
            self.T = image_head_sd = text_head_sd = None 
            from_scratch = True
        return from_scratch, image_head_sd, text_head_sd, model 

    def _build_siamese_backbone(self, **kwargs):
        from_scratch, image_head_sd, text_head_sd, _ = self._load_clip(None) 

        self.image_head = build_image_head(self.cfg.model.image)
        if not from_scratch and not self.cfg.model.image.from_scratch:
            self.image_head.copy_state_dict(image_head_sd)
            self.echo("Initialize image encoder from `image_head`.")

        self.audio_head = build_audio_head(self.cfg.model.audio)
        self.audio_head.encoder = self.image_head.encoder

        self.text_head = build_text_head(self.cfg.model.text)
        self.text_head.encoder = self.image_head.encoder

        self.loss_head = build_loss_head(self.cfg.model.loss)

        tunable_params = {
            f"loss_head.{k}": v for k, v in self.loss_head.named_parameters()
        } 
        if not self.cfg.model.image.freeze:
            tunable_params.update({
                f"text_head.{k}": v for k, v in self.text_head.named_parameters()
            })
        else:
            self.echo("Freeze image/audio/text encoder.")
        return tunable_params

    def _build_separate_backbone(self, **kwargs):
        from_scratch, image_head_sd, text_head_sd, _ = self._load_clip(None) 
            
        self.image_head = build_image_head(self.cfg.model.image)
        if not from_scratch and not self.cfg.model.image.from_scratch:
            self.image_head.copy_state_dict(image_head_sd)
            self.echo("Initialize image encoder from `image_head`.")

        self.audio_head = build_audio_head(self.cfg.model.audio)
        if not from_scratch and not self.cfg.model.audio.from_scratch:
            self.audio_head.copy_state_dict(image_head_sd)
            self.echo("Initialize audio encoder from `image_head`.")

        self.text_head = build_text_head(self.cfg.model.text)
        if not from_scratch and not self.cfg.model.text.from_scratch:
            self.text_head.copy_state_dict(text_head_sd)
            self.echo("Initialize text encoder from `text_head`.")

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
        if not self.cfg.model.image.freeze:
            tunable_params.update({
                f"image_head.{k}": v for k, v in self.image_head.named_parameters()
            })
        else:
            self.echo("Freeze image encoder.")
        if not self.cfg.model.text.freeze:
            tunable_params.update({
                f"text_head.{k}": v for k, v in self.text_head.named_parameters()
            })
        else:
            self.echo("Freeze text encoder.")
        return tunable_params

