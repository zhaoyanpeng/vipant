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

from ..util import make_dummy_image_with_text
from ..module import (
    build_image_head, build_audio_head, build_text_head, build_loss_head
)
from . import (
    load_checkpoint, load_clip, load_meme
)

from mreserve.preprocess import video_to_segments, preprocess_video, encoder, MASK
from mreserve.preprocess import preprocess_image_to_patches, AUDIOSPAN, MASKAUDIO
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
        elif encoder_type == "audio_only":
            audio_features = self.model.embed_audio_only(videos["audio_clips"])
            audio_features = audio_features[None]
        else:
            raise ValueError(f"Unsupported audio encoder type `{encoder_type}`.")
        loss = self.loss_head(audio_features, labels, **kwargs)
        return loss     
    
    def encode_text(self, text, *args, device_ids=[0], **kwargs):
        """ text is a list of str.
        """
        if self.cfg.running.visual_prompt:
            label_map = kwargs.get("label_map", None)
            return self.encode_visual_prompt_text(text, *args, label_map=label_map)
        text_features = self.model.get_label_space(text)
        return text_features

    def encode_visual_prompt_text(self, text, *args, device_ids=[0], label_map=None, **kwargs):
        """ visual image as a prompt.
        """
        text_features = []
        grid_size = tuple(self.cfg.running.audio.grid_size)
        font = "/usr/share/fonts/truetype/lato/Lato-Regular.ttf"
        if label_map is None or not isinstance(label_map[0], str):
            images = [make_dummy_image_with_text(name, font, grid_size) for name in text]
            #images = [make_dummy_image_with_text(
            #    re.sub(f"^{self.cfg.running.prompt}", "", name).strip(),
            #font, grid_size) for name in text]
        else:
            images = [Image.open(label_map[i]) for i in range(len(text))]
        #images = np.stack(images)
        for i, name in enumerate(text):
            img_i = np.array(images[i])
            img_i_proc = preprocess_image_to_patches(img_i, output_grid_size=grid_size)
            txt = encoder.encode(text[i]).ids # [] #

            mask_idx = len(txt) + 1
            subseg_idxs = [0] * len(txt)
            txt.append(MASK)
            txt.append(MASKAUDIO)
            subseg_idxs.append(3)
            subseg_idxs.append(3)

            out = self.model.embed_video(
                np.tile(img_i_proc[None], [2, 1, 1]),
                audio_clips=np.zeros([6, 60, 65], dtype=np.float32),
                tokens=np.array(txt, dtype=np.int32),
                subseg_idxs=np.array(subseg_idxs, dtype=np.int32)
            )
            audio_pred = out[mask_idx]
            text_features.append(audio_pred)
            #print(f"{i}-th label done.", end="\n")
        text_features = np.stack(text_features)
        #self.build(**{"output_dim": len(text)})
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
