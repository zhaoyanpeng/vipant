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
from cvap.util import unit_normalize

from ..module import (
    build_image_head, build_audio_head, build_text_head, build_loss_head
)
from . import (
    load_checkpoint, load_clip, load_meme
)

from mreserve.preprocess import video_to_segments, preprocess_video, encoder, MASK
from mreserve.modeling import PretrainedMerlotReserve
import jax
import jax.numpy as jnp

class MreserveClassifier(nn.Module):
    def __init__(self, cfg, echo):
        super(MreserveClassifier, self).__init__()
        self.cfg = cfg
        self.echo = echo

    def forward(self, audios, labels, *args, videos=None, **kwargs):
        # how to asynchronize the two `data_parallel` 
        kwargs = {"normalized": True, "names": kwargs.get("names", None)}
        encoder_type = self.cfg.running.encoder_type
        if encoder_type == "audio":
            audio_features = self.model.embed_unnormalized_audio(videos["audio_clips"])
            audio_features = jnp.mean(audio_features, 0, keepdims=True) 
            audio_features = unit_normalize(audio_features)
        elif encoder_type == "joint":
            out_h = self.model.embed_video(**videos)
            audio_features = out_h[videos['tokens'] == MASK]
        else:
            raise ValueError(f"Unsupported audio encoder type `{encoder_type}`.")
        loss = self.loss_head(audio_features, labels, **kwargs)
        return loss     
    
    def encode_text(self, text, *args, device_ids=[0], **kwargs):
        """ text is a list of str.
        """
        text_features = self.model.get_label_space(text)
        return text_features

    def collect_audio_state_dict(self):
        return dict() 

    def report(self, gold_file=None, **kwargs):
        if not dist.is_initialized() or dist.get_rank() == 0:
            return self.loss_head.report(gold_file=gold_file, **kwargs)
        else:
            return ""
    
    def build(self, **kwargs):
        tunable_params = dict()
        if self.cfg.eval:
            # This handles loading the model and getting the checkpoints.
            grid_size = tuple(self.cfg.running.audio.grid_size)
            self.model = PretrainedMerlotReserve.from_pretrained(
                model_name='large', image_grid_size=grid_size
            )
            self.loss_head = build_loss_head(self.cfg.model.loss, **kwargs)
        else:
            raise ValueError("Unsupported learning")
        return tunable_params
