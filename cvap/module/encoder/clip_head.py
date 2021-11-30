from fvcore.common.registry import Registry
from omegaconf.listconfig import ListConfig
from collections import OrderedDict

import re
import math
import copy
import threading
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .. import (
    build_encoder_module, interp_clip_vp_embedding, interp_conv_weight_spatial
)
from .audio_head import position_resolution, load_pos_embedding

""" The idea is to abstract an encoding head as a four-layer encoder. 
    (1) backbone encoder (most likely to be shared)
    (2-3) modality-specific pre- / post-encoding layer
    (4) class / positional embedding (likely to be shared) 
"""

class MetaHead(nn.Module):
    def __init__(self, cfg, **kwargs):
        super().__init__()
        keep_hp = kwargs.pop("keep_hp", False)
        reference = kwargs.pop("reference", None)
        shared_modules = kwargs.pop("shared_modules", [])
        kwargs.update({
            "width": cfg.width, "embed_dim": cfg.embed_dim, 
            "ctx_len": cfg.ctx_len, "resolution": cfg.resolution
        }) # shared hyperparameters

        self.encoder = (
            build_encoder_module(cfg.encoder, **kwargs) 
            #if "encoder" not in shared_modules else reference.encoder 
        ) # backbone
        self.pre_encoder = (
            build_encoder_module(cfg.pre_encoder, **kwargs)
            #if "pre_encoder" not in shared_modules else reference.pre_encoder 
        )
        self.post_encoder = (
            build_encoder_module(cfg.post_encoder, **kwargs)
            #if "post_encoder" not in shared_modules else reference.post_encoder 
        )

        self.pre_encoder_addon = build_encoder_module(
            cfg.pre_encoder_addon, **kwargs
        ) # in-between `pre_encoder` & `encoder`
        self.post_encoder_addon = build_encoder_module(
            cfg.post_encoder_addon, **kwargs
        ) # in-between `encoder` & `post_encoder`

        # have to build all modules to get `position_resolution`, even though 
        # we will probably replace all the modules by those of the `reference` 
        position_resolution = (
            self.pre_encoder.position_resolution or \
            self.encoder.position_resolution or \
            self.post_encoder.position_resolution
        ) 
        kwargs.update({
            "position_resolution": position_resolution
        })
        self.misc = build_encoder_module(cfg.misc, **kwargs)

        # time to share modules
        #self.replace_modules(shared_modules, reference, keep_hp=keep_hp)

    def replace_modules(self, shared_modules=[], reference=None, keep_hp=False, **kwargs):
        """ keep_hp: keep selected hyperparameters
        """
        if len(shared_modules) < 1 or reference is None:
            return []
        module_list = ["encoder", "pre_encoder", "post_encoder", "misc"]
        ref_modules = list()
        for module in module_list:
            if module not in shared_modules: 
                continue
            ref_modules.append(module)
            self_module = eval(f"self.{module}")
            refr_module = eval(f"reference.{module}")
            #print(f"RP A {module} {self_module.hp} {refr_module.hp} {self_module == refr_module}")
            if hasattr(self_module, "replace_modules"):
                self_module.replace_modules(refr_module, keep_hp=keep_hp)
                new_self_module = eval(f"self.{module}")
                #print(f"RP B {module} {self_module.hp} {refr_module.hp} {self_module == refr_module} {new_self_module == refr_module}")
            else: # via reference, not recommended
                hp = self_module.hp 
                exec(f"self.{module} = reference.{module}") # modified via reference
                if keep_hp:
                    exec(f"self.{module}.hp = {hp}") # so the `reference` is modified
                new_self_module = eval(f"self.{module}")
                #print(f"RP C {module} {self_module.hp} {refr_module.hp} {self_module == refr_module} {new_self_module == refr_module}")
        return ref_modules

    def forward(self, x: torch.Tensor, *args, **kwargs):
        kwargs.update({
            "positional_embedding": self.misc.pos_embedding, 
            "class_embedding": self.misc.cls_embedding,
            "position_resolution": self.misc.position_resolution
        })
        x = self.pre_encoder(x, **kwargs) # (N, L, D)
        x = self.pre_encoder_addon(x, **kwargs) # (N, L, D)
        
        # TODO assumed 3d `x`
        x = x.permute(1, 0, 2) if not self.encoder.batch_first else x # (N, L, D) -> (L, N, D)
        x = self.encoder(x, **kwargs) 
        x = x.permute(1, 0, 2) if not self.encoder.batch_first else x # (L, N, D) -> (N, L, D)

        mask = self.pre_encoder.mask #or self.encoder.mask # text) postion of cls token; audio/image) ?

        x = self.post_encoder_addon(x, **kwargs) 
        x = self.post_encoder(x, mask=mask, **kwargs)

        if kwargs.get("normalized", False):
            x = x / x.norm(dim=-1, keepdim=True)
            #print(f"{threading.current_thread().ident} x --{kwargs.get('normalized', False)}")
        return x

class CLIPImageHead(MetaHead):
    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)
    
    def copy_state_dict(self, state_dict):
        if not self.encoder.batch_first: # TransformerBackbone
            pre_keys = {"conv1.weight"}
            post_keys = {"proj"}
            misc_keys = {"positional_embedding", "class_embedding"}
            old_dict = OrderedDict()
            for k, v in state_dict.items():
                if k in pre_keys:
                    k = f"pre_encoder.{k}" 
                elif k in post_keys:
                    k = f"post_encoder.{k}"
                elif k in misc_keys:
                    k = f"misc.{k}"
                else:
                    #k = re.sub("^ln_\w+\.", "ln.", k)
                    k = re.sub("^transformer\.", "encoder.", k)
                    k = re.sub("^ln_pre\.", "pre_encoder.ln.", k)
                    k = re.sub("^ln_post\.", "post_encoder.ln.", k)
                old_dict[k] = v
        else: # ResNetBackbone
            old_dict = OrderedDict()
            for k, v in state_dict.items():
                if re.match("layer\d+\.", k):
                    k = f"encoder.{k}"
                elif re.match("attnpool\.", k):
                    k = re.sub("^attnpool\.", "post_encoder.", k)
                else:
                    k = f"pre_encoder.{k}"
                old_dict[k] = v
            pos_key = "post_encoder.positional_embedding" 
            new_key = "misc." + pos_key.rsplit(".")[-1]
            old_dict[new_key] = old_dict.pop(pos_key)
        new_dict = self.state_dict()
        new_keys = set(new_dict.keys())
        old_keys = set(old_dict.keys())
        new_dict.update(old_dict)
        self.load_state_dict(new_dict)
        n_o = new_keys - old_keys
        o_n = old_keys - new_keys
        #print(f"{n_o}\n{o_n}")
        return n_o, o_n

class CLIPAudioHead(MetaHead):
    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)

    def from_pretrained(self, state_dict, cfg, *args, **kwargs):
        excluded = ["misc.positional_embedding"]
        new_dict = self.state_dict()
        old_dict = {k: v for k, v in state_dict.items() if k not in excluded}
        # interpolate positional embedding
        key = "misc.positional_embedding"
        new_pos_shape = self.misc.position_resolution
        old_pos_shape = position_resolution(
            cfg.model.audio.resolution, cfg.model.audio.pre_encoder.patch_size, cfg.model.audio.pre_encoder.stride
        ) # nrow always indicates the time dimenstion
        #print(new_dict[key].shape, state_dict[key].shape, new_pos_shape, old_pos_shape)
        if state_dict[key].shape[0] in {50, 197}: # from vision encoder TODO could be wrong
            state_dict[key] = interp_clip_vp_embedding(
                state_dict.pop(key), old_pos_shape
            ) # pos embed inherited from vision encoder
        n_o, o_n = load_pos_embedding(
            state_dict, old_dict, new_dict, key, 1, old_pos_shape, new_pos_shape
        )
        self.load_state_dict(new_dict)
        return n_o, o_n

    def copy_state_dict(self, state_dict):
        if not self.encoder.batch_first: # TransformerBackbone
            pre_keys = {"conv1.weight"}
            post_keys = {"proj"}
            misc_keys = {"positional_embedding", "class_embedding"}
            old_dict = OrderedDict()
            for k, v in state_dict.items():
                if k in pre_keys:
                    k = f"pre_encoder.{k}" 
                elif k in post_keys:
                    k = f"post_encoder.{k}"
                elif k in misc_keys:
                    k = f"misc.{k}"
                else:
                    #k = re.sub("^ln_\w+\.", "ln.", k)
                    k = re.sub("^transformer\.", "encoder.", k)
                    k = re.sub("^ln_pre\.", "pre_encoder.ln.", k)
                    k = re.sub("^ln_post\.", "post_encoder.ln.", k)
                old_dict[k] = v
            # interpolation
            pos_key = "misc.positional_embedding"
            old_dict[pos_key] = interp_clip_vp_embedding(
                old_dict.pop(pos_key), self.misc.position_resolution
            )
        else: # ResNetBackbone
            old_dict = OrderedDict()
            for k, v in state_dict.items():
                if re.match("layer\d+\.", k):
                    k = f"encoder.{k}"
                elif re.match("attnpool\.", k):
                    k = re.sub("^attnpool\.", "post_encoder.", k)
                else:
                    k = f"pre_encoder.{k}"
                old_dict[k] = v
            # interpolation
            pos_key = "post_encoder.positional_embedding" 
            new_key = "misc." + pos_key.rsplit(".")[-1]
            old_dict[new_key] = interp_clip_vp_embedding(
                old_dict.pop(pos_key), self.misc.position_resolution
            )
        # take care of conv1
        new_dict = self.state_dict()
        conv_key = "pre_encoder.conv1.weight"
        conv_weight = interp_conv_weight_spatial(old_dict[conv_key], new_dict[conv_key].shape[-2:])
        use_mean = new_dict[conv_key].shape[1] != 1
        old_dict[conv_key] = conv_weight if use_mean else conv_weight.mean(1, keepdim=True)
        # update
        new_keys = set(new_dict.keys())
        old_keys = set(old_dict.keys())
        new_dict.update(old_dict)
        self.load_state_dict(new_dict)
        n_o = new_keys - old_keys
        o_n = old_keys - new_keys
        #print(f"{n_o}\n{o_n}")
        return n_o, o_n

class CLIPTextHead(MetaHead):
    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)
        self.initialize_parameters()

    def initialize_parameters(self):
        pass #nn.init.normal_(self.positional_embedding, std=0.01)
    
    def copy_state_dict(self, state_dict):
        pre_keys = {"token_embedding.weight"}
        post_keys = {}
        misc_keys = {"positional_embedding"}
        old_dict = OrderedDict()
        for k, v in state_dict.items():
            if k in pre_keys:
                k = f"pre_encoder.{k}" 
            elif k in post_keys:
                k = f"post_encoder.{k}"
            elif k in misc_keys:
                k = f"misc.{k}"
            else:
                #k = re.sub("^ln_\w+\.", "ln.", k)
                k = re.sub("^transformer\.", "encoder.", k)
                k = re.sub("^ln_final\.", "post_encoder.ln.", k)
                k = re.sub("^text_projection", "post_encoder.proj", k)
            old_dict[k] = v
        new_dict = self.state_dict()
        # TODO better via interpolation
        pos_key = "misc.positional_embedding"
        old_num = old_dict[pos_key].shape[0]
        new_num = new_dict[pos_key].shape[0]
        if old_num >= new_num:
            old_dict[pos_key] = old_dict.pop(pos_key)[:new_num]
        else:
            new_dict[pos_key][:old_num] = old_dict.pop(pos_key)
            old_dict[pos_key] = new_dict[pos_key] # unnecessary
        new_keys = set(new_dict.keys())
        old_keys = set(old_dict.keys())
        new_dict.update(old_dict)
        self.load_state_dict(new_dict)
        n_o = new_keys - old_keys
        o_n = old_keys - new_keys
        #print(f"{n_o}\n{o_n}")
        return n_o, o_n
