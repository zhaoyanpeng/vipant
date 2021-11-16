from omegaconf import OmegaConf
import os, re
import torch
from torch import nn

import torch.distributed as dist
from torch.nn.parallel import data_parallel
from collections import defaultdict, OrderedDict

from ..module import (
    build_image_head, build_audio_head, build_text_head, build_loss_head
)
from . import (
    load_checkpoint, load_clip, load_meme
)

from clip import load 

class CLVPDP(nn.Module):
    def __init__(self, cfg, echo):
        super(CLVPDP, self).__init__()
        self.cfg = cfg
        self.echo = echo

    def forward(self, images, text, *args, **kwargs):
        if kwargs.get("retrieval", False): # if it is a retrieval task
            return self.forward_retrieval(images, text, *args, **kwargs)
        else:
            raise ValueError("Only support retrieval.")

    def forward_retrieval(self, images, text, *args, **kwargs):
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
        text_features = data_parallel(
            self.text_head, text, device_ids=device_ids, module_kwargs=kwargs
        )
        loss = self.loss_head(image_features, text_features, **kwargs)
        return loss

    def collect_audio_state_dict(self):
        return (dict(),) * 2 

    def collect_state_dict(self):
        return (dict(),) * 3

    def report(self, gold_file=None):
        if not dist.is_initialized() or dist.get_rank() == 0:
            return self.loss_head.report(gold_file=gold_file)
        else:
            return ""
    
    def build(self, **kwargs):
        tunable_params = dict()
        if self.cfg.eval:
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

            self.text_head = build_text_head(self.cfg.model.text) #
            n_o, o_n = self.text_head.copy_state_dict(text_head_sd)
            msg = f" except {n_o}" if len(n_o) > 0 else ""
            self.echo(f"Initialize text encoder from `text_head`{msg}.")

            self.loss_head = build_loss_head(self.cfg.model.loss, **kwargs)
            if loss_head_sd is not None:
                self.loss_head.copy_state_dict(loss_head_sd) #

            self.cuda(self.cfg.rank) 
        else:
            raise ValueError("Not implemented yet.")
