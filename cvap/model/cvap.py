from omegaconf import OmegaConf
import os, re
import torch
from torch import nn

import torch.distributed as dist
from torch.nn.parallel import data_parallel

from ..module import (
    build_image_head, build_audio_head, build_text_head, build_loss_head
)
from . import (
    load_checkpoint, load_clip, load_meme
)

from clip import load 

class CVAP(nn.Module):
    def __init__(self, cfg, echo):
        super(CVAP, self).__init__()
        self.cfg = cfg
        self.echo = echo

    def forward(self, images, audios, *args, **kwargs):
        device_ids = kwargs.get("device_ids", [0])
        # how to asynchronize the two `data_parallel` 
        kwargs = {"normalized": self.loss_head.normalized, "names": kwargs.get("names", None)}
        if self.image_head is not None:
            image_features = data_parallel(
                self.image_head, images, device_ids=device_ids, module_kwargs=kwargs
            )
        else: # pre-computed unnormalized features
            if self.loss_head.normalized:
                images = images / images.norm(dim=-1, keepdim=True)
            image_features = images
        audio_features = data_parallel(
            self.audio_head, audios, device_ids=device_ids, module_kwargs=kwargs
        )
        loss = self.loss_head(image_features, audio_features, **kwargs)
        return loss     

    def collect_audio_state_dict(self):
        return (
            self.audio_head.state_dict(), 
            self.loss_head.state_dict(),
        )

    def collect_state_dict(self):
        return (
            self.image_head.state_dict(), 
            self.audio_head.state_dict(), 
            self.loss_head.state_dict(),
        )

    def report(self, gold_file=None):
        if not dist.is_initialized() or dist.get_rank() == 0:
            return self.loss_head.report(gold_file=gold_file)
        else:
            return ""
    
    def build(self):
        tunable_params = dict()
        if self.cfg.eval:
            local_cfg, _, audio_head_sd, _, loss_head_sd = load_checkpoint(self.cfg, self.echo)
            from_scratch, image_head_sd, _, _ = load_clip(None, self.cfg, self.echo)

            self.image_head = build_image_head(self.cfg.model.image)
            self.image_head.copy_state_dict(image_head_sd)

            self.audio_head = build_audio_head(local_cfg.model.audio)
            self.audio_head.load_state_dict(audio_head_sd)

            self.loss_head = build_loss_head(local_cfg.model.loss)
            self.loss_head.load_state_dict(loss_head_sd)
            self.cuda(self.cfg.rank) 
        else:
            # try pre-trained model!
            local_cfg, _, audio_head_sd, _, loss_head_sd = load_checkpoint(self.cfg, self.echo)
            # try clip! TODO do we always have to load CLIP?
            from_scratch, image_head_sd, _, model = load_clip(local_cfg, self.cfg, self.echo)
            # try meme!
            with_meme, meme_image_head_sd = load_meme(self.cfg, self.echo)

            self.image_head = build_image_head(self.cfg.model.image)
            if not from_scratch and not self.cfg.model.image.from_scratch:
                self.image_head.copy_state_dict(image_head_sd)
                self.echo("Initialize image encoder from `image_head`.")
            if self.cfg.running.frame_emb is not None:
                self.image_head = None
                self.echo("Destory image encoder.")

            self.audio_head = build_audio_head(self.cfg.model.audio)
            if not self.cfg.model.audio.from_scratch:
                if local_cfg is not None:
                    # TODO better to use `from_pretrained()`
                    self.audio_head.load_state_dict(audio_head_sd)
                    self.echo("Initialize audio encoder from `audio_head`.")
                elif not from_scratch:
                    if with_meme: # higher priority
                        msg = " `meme_image_head`"
                        n_o, o_n = self.audio_head.copy_state_dict(meme_image_head_sd)
                    else:
                        msg = " `image_head`"
                        n_o, o_n = self.audio_head.copy_state_dict(image_head_sd)
                    msg += f" except {n_o}" if len(n_o) > 0 else ""
                    self.echo(f"Initialize audio encoder from{msg}.")
                else:
                    self.echo("Have to learn from scratch.")
                
            self.loss_head = build_loss_head(self.cfg.model.loss)
            if not from_scratch and not self.cfg.model.audio.from_scratch:
                extra_sd = {"logit_scale": model.logit_scale}
                self.loss_head.copy_state_dict(extra_sd)

            tunable_params = {
                f"audio_head.{k}": v for k, v in self.audio_head.named_parameters()
            } 
            tunable_params.update({
                f"loss_head.{k}": v for k, v in self.loss_head.named_parameters()
            })
            if not self.cfg.model.image.freeze and self.image_head is not None:
                tunable_params.update({
                    f"image_head.{k}": v for k, v in self.image_head.named_parameters()
                })
            elif self.image_head is not None:
                self.echo("Freeze image encoder.")
            self.cuda(self.cfg.rank)
        return tunable_params

