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

class CVAPDDP(nn.Module):
    def __init__(self, cfg, echo):
        super(CVAPDDP, self).__init__()
        self.cfg = cfg
        self.echo = echo

    def forward(self, images, audios, *args, **kwargs):
        # use gather or reduce, that depends on the loss_head
        image_features = self.image_head(images)
        audio_features = self.audio_head(audios)
        
        if not self.loss_head.reduce:
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            audio_features = audio_features / audio_features.norm(dim=-1, keepdim=True)

            image_list = [torch.zeros_like(image_features) for _ in range(self.cfg.num_gpus)]
            audio_list = [torch.zeros_like(audio_features) for _ in range(self.cfg.num_gpus)]

            dist.all_gather(tensor_list=image_list, tensor=image_features.contiguous())
            dist.all_gather(tensor_list=audio_list, tensor=audio_features.contiguous())

            image_list[dist.get_rank()] = image_features 
            audio_list[dist.get_rank()] = audio_features 

            image_features = torch.cat(image_list)
            audio_features = torch.cat(audio_list)

            loss = self.loss_head(image_features, audio_features, normalized=True)
        else: # TODO barlow loss
            pass
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

    def report(self):
        if dist.get_rank() == 0:
            return self.loss_head.report()
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
            model, self.T = load(
                rcfg.clip_model_name, rcfg.clip_model_root, device="cpu", jit=False
            )
            image_head_sd = model.visual.state_dict()
            self.image_head = build_image_head(self.cfg.model.image)
            self.image_head.copy_state_dict(image_head_sd)

            self.audio_head = build_audio_head(self.cfg.model.audio)
            self.audio_head.copy_state_dict(image_head_sd)
                
            extra_sd = {"logit_scale": model.logit_scale}
            self.loss_head = build_loss_head(self.cfg.model.loss)
            self.loss_head.copy_state_dict(extra_sd)

            tunable_params = {
                f"audio_head.{k}": v for k, v in self.audio_head.named_parameters()
            } 
            tunable_params.update({
                f"loss_head.{k}": v for k, v in self.loss_head.named_parameters()
            })
            self.cuda(self.cfg.rank)
        return tunable_params

