from omegaconf import OmegaConf
import os, re
import torch
from collections import OrderedDict

from clip import load 

__all__ = ["load_checkpoint", "load_clip", "load_meme", "extract_model_file"]

def load_checkpoint(cfg, echo):
    model_file = f"{cfg.model_root}/{cfg.model_name}/{cfg.model_file}"
    try:
        checkpoint = torch.load(model_file, map_location="cpu")
        echo(f"Loading from {model_file}")
    except Exception as e:
        echo(f"Failed to load the checkpoint `{model_file}`")
        return (None,) * 5
    local_cfg = checkpoint["cfg"]
    local_str = OmegaConf.to_yaml(local_cfg)
    if cfg.verbose:
        echo(f"Old configs:\n\n{local_str}")
    nmodule = len(checkpoint["model"])
    if nmodule == 2:
        audio_head_sd, loss_head_sd = checkpoint["model"]
        return local_cfg, None, audio_head_sd, None, loss_head_sd
    elif nmodule == 4:
        image_head_sd, audio_head_sd, text_head_sd, loss_head_sd = checkpoint["model"]
        return local_cfg, image_head_sd, audio_head_sd, text_head_sd, loss_head_sd
    else:
        raise ValueError(f"I don't know how to parse the checkpoint: # module is {nmodule}.")

def load_clip(local_cfg, cfg, echo):
    try: # try image / text backbone
        rcfg = cfg.running
        model, _ = load(
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
        echo(f"Will learn from scratch because: {e}") 
        model = image_head_sd = text_head_sd = None
        from_scratch = True
    return from_scratch, image_head_sd, text_head_sd, model 

def load_meme(cfg, echo):
    try: # try image / text backbone
        acfg = cfg.model.audio
        model = torch.hub.load(acfg.meme_path, acfg.meme_name, pretrained=True)
        image_head_sd = model.state_dict()
        with_meme = True 
    except Exception as e:
        meme_name = getattr(acfg, "meme_name", None)
        echo(f"Failed to load the meme `{meme_name}` because: {e}")
        image_head_sd = None 
        with_meme = False 
    return with_meme, image_head_sd

def extract_model_file(cfg, echo):
    model_files = list()
    log_file = f"{cfg.model_root}/{cfg.model_name}/{cfg.model_file}"
    try:
        pattern = '.?Saving the checkpoint to.*\/([0-9]+\.pth)'
        with open(log_file, "r") as fr:
            for line in fr:
                ret = re.search(pattern, line)
                if ret is not None:
                    model_files.append(ret.groups(0)[0])
    except Exception as e:
        echo(f"Failed to extract model files from `{log_file}`")
    return model_files
