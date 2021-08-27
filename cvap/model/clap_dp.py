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

from clip import load 

class CLAPDP(nn.Module):
    def __init__(self, cfg, echo):
        super(CLAPDP, self).__init__()
        self.cfg = cfg
        self.echo = echo

    def forward(self, audios, text, *args, **kwargs):
        if kwargs.get("retrieval", False): # if it is a retrieval task
            return self.forward_retrieval(audios, text, *args, **kwargs)
        device_ids = kwargs.get("device_ids", [0])
        # how to asynchronize the two `data_parallel`
        kwargs = {"normalized": False, "names": kwargs.get("names", None)}
        _, audio_features = data_parallel(
            self.audio_head, audios, device_ids=device_ids, module_kwargs=kwargs
        )
        text_input = (text, audio_features, self.audio_head.time_first)
        _, logits, predictions = data_parallel(
            self.text_head, text_input, device_ids=device_ids, module_kwargs=kwargs
        )
        loss = self.loss_head(logits, text[:, 1:], predictions, **kwargs)
        return loss  

    def forward_retrieval(self, audios, text, *args, **kwargs):
        device_ids = kwargs.get("device_ids", [0])
        # how to asynchronize the two `data_parallel`
        kwargs = {"normalized": True, "names": kwargs.get("names", None)}
        audio_features = data_parallel(
            self.audio_head, audios, device_ids=device_ids, module_kwargs=kwargs
        )
        text_features = data_parallel(
            self.text_head, text, device_ids=device_ids, module_kwargs=kwargs
        )
        loss = self.loss_head(audio_features, text_features, **kwargs)
        return loss

    def collect_audio_state_dict(self):
        return (
            self.audio_head.state_dict(), 
            self.loss_head.state_dict(),
        )

    def collect_state_dict(self):
        return (
            self.audio_head.state_dict(), 
            self.text_head.state_dict(), 
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
        def load_clip(local_cfg):
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
        if self.cfg.eval:
            local_cfg, audio_head_sd, loss_head_sd = load_checkpoint()
            from_scratch, _, text_head_sd, _ = load_clip(local_cfg) 

            self.audio_head = build_audio_head(local_cfg.model.audio)
            self.audio_head.load_state_dict(audio_head_sd)

            self.text_head = build_text_head(self.cfg.model.text) #
            self.text_head.copy_state_dict(text_head_sd)

            self.loss_head = build_loss_head(self.cfg.model.loss)
            self.loss_head.load_state_dict(loss_head_sd)
            self.cuda(self.cfg.rank) 
        else:
            # try pre-trained model
            local_cfg, audio_head_sd, loss_head_sd = load_checkpoint()
            from_scratch, image_head_sd, text_head_sd, model = load_clip(local_cfg) 

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
                    self.audio_head.from_pretrained(audio_head_sd, local_cfg)
                    self.echo("Initialize audio encoder from `audio_head`.")
                elif not from_scratch:
                    self.audio_head.copy_state_dict(image_head_sd)
                    self.echo("Initialize audio encoder from `image_head`.")
                else:
                    self.echo("Have to learn from scratch.")

            self.text_head = build_text_head(self.cfg.model.text)
            if not from_scratch:
                self.text_head.copy_state_dict(text_head_sd)
                self.echo("Initialize text encoder from `text_head`.")
                
            self.loss_head = build_loss_head(self.cfg.model.loss)
            if not from_scratch and not self.cfg.model.audio.from_scratch:
                extra_sd = {"logit_scale": model.logit_scale}
                self.loss_head.copy_state_dict(extra_sd)

            tunable_params.update({
                f"loss_head.{k}": v for k, v in self.loss_head.named_parameters()
            })
            if not self.cfg.model.audio.freeze:
                tunable_params.update({
                    f"audio_head.{k}": v for k, v in self.audio_head.named_parameters()
                })
            else:
                self.echo("Freeze audio encoder.")
            if not self.cfg.model.text.freeze:
                tunable_params.update({
                    f"text_head.{k}": v for k, v in self.text_head.named_parameters()
                })
            else:
                self.echo("Freeze text encoder.")
            self.cuda(self.cfg.rank)
        return tunable_params

