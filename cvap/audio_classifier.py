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

from .module import (
    build_image_head, build_audio_head, build_text_head, build_loss_head
)


class AudioClassifier(nn.Module):
    def __init__(self, cfg, echo):
        super(AudioClassifier, self).__init__()
        self.cfg = cfg
        self.echo = echo

    def forward(self, audios, labels, *args, **kwargs):
        device_ids = kwargs.get("device_ids", [0])
        # how to asynchronize the two `data_parallel` 
        kwargs = {"normalized": False, "names": kwargs.get("names", None)}
        audio_features = data_parallel(
            self.audio_head, audios, device_ids=device_ids, module_kwargs=kwargs
        )
        loss = self.loss_head(audio_features, labels, **kwargs)
        return loss     

    def collect_audio_state_dict(self):
        return (
            self.audio_head.state_dict(), 
            self.loss_head.state_dict(),
        )

    def report(self, gold_file=None):
        if not dist.is_initialized() or dist.get_rank() == 0:
            return self.loss_head.report(gold_file=gold_file)
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
            audio_head_sd, loss_head_sd = checkpoint["model"]
            return local_cfg, audio_head_sd, loss_head_sd 

        if self.cfg.eval:
            local_cfg, audio_head_sd, loss_head_sd = load_checkpoint()

            self.audio_head = build_audio_head(local_cfg.model.audio, **kwargs)
            self.audio_head.load_state_dict(audio_head_sd)

            self.loss_head = build_loss_head(local_cfg.model.loss, **kwargs)
            self.loss_head.load_state_dict(loss_head_sd)

            self.cuda(self.cfg.rank) 
        else:
            # try pre-trained model
            local_cfg, audio_head_sd, loss_head_sd = load_checkpoint()
            if local_cfg is None: # try image backbone
                rcfg = self.cfg.running
                try:
                    model, self.T = load(
                        rcfg.clip_model_name, rcfg.clip_model_root, device="cpu", jit=False
                    )
                    image_head_sd = model.visual.state_dict()
                    from_scratch = False
                except Exception as e:
                    self.echo(f"Will learn from scratch because: {e}") 
                    image_head_sd = None 
                    from_scratch = True
            
            #cfg = local_cfg if local_cfg is not None else self.cfg
            self.audio_head = build_audio_head(self.cfg.model.audio, **kwargs)
            if not self.cfg.model.audio.from_scratch:
                if local_cfg is not None:
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
