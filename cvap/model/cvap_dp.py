from omegaconf import OmegaConf
import os, re
import torch
from torch import nn

import torch.distributed as dist
from torch.nn.parallel import data_parallel

from ..module import (
    build_image_head, build_audio_head, build_text_head, build_loss_head
)

from clip import load 

class CVAPDP(nn.Module):
    def __init__(self, cfg, echo):
        super(CVAPDP, self).__init__()
        self.cfg = cfg
        self.echo = echo

    def forward(self, images, audios, *args, **kwargs):
        device_ids = kwargs.get("device_ids", [0])
        # how to asynchronize the two `data_parallel` 
        kwargs = {"normalized": True, "names": kwargs.get("names", None)}
        image_features = data_parallel(
            self.image_head, images, device_ids=device_ids, module_kwargs=kwargs
        )
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
            model_file = f"{self.cfg.model_root}/{self.cfg.model_name}/{self.cfg.model_file}"
            self.echo(f"Loading from {model_file}")
            checkpoint = torch.load(model_file, map_location="cpu")
            local_cfg = checkpoint["cfg"]
            local_str = OmegaConf.to_yaml(local_cfg)
            self.echo(f"Old configs:\n\n{local_str}")
            audio_head_sd, loss_head_sd = checkpoint["model"]

            rcfg = local_cfg.running
            model, self.T = load(
                rcfg.clip_model_name, rcfg.clip_model_root, device="cpu", jit=False
            )
            image_head_sd = model.visual.state_dict()
            self.image_head = build_image_head(local_cfg.model.image)
            self.image_head.copy_state_dict(image_head_sd)

            self.audio_head = build_audio_head(local_cfg.model.audio)
            self.audio_head.load_state_dict(audio_head_sd)

            self.loss_head = build_loss_head(local_cfg.model.loss)
            self.loss_head.load_state_dict(loss_head_sd)
            self.cuda(self.cfg.rank) 
        else:
            rcfg = self.cfg.running
            try:
                model, self.T = load(
                    rcfg.clip_model_name, rcfg.clip_model_root, device="cpu", jit=False
                )
                image_head_sd = model.visual.state_dict()
                from_scratch = False
            except Exception as e:
                self.echo(f"Will learn from scratch because: {e}") 
                from_scratch = True
            self.image_head = build_image_head(self.cfg.model.image)
            if not from_scratch and not self.cfg.model.image.from_scratch:
                self.image_head.copy_state_dict(image_head_sd)
                self.echo("Initialize image encoder from `image_head`.")

            self.audio_head = build_audio_head(self.cfg.model.audio)
            if not from_scratch and not self.cfg.model.audio.from_scratch:
                self.audio_head.copy_state_dict(image_head_sd)
                self.echo("Initialize audio encoder from `image_head`.")
                
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
            if not self.cfg.model.image.freeze:
                tunable_params.update({
                    f"image_head.{k}": v for k, v in self.image_head.named_parameters()
                })
            else:
                self.echo("Freeze image encoder.")
            self.cuda(self.cfg.rank)
        return tunable_params

