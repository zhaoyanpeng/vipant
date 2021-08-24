import os
import re
import glob
import json
import torch
import random
import warnings
import itertools
import torchaudio
import numpy as np
import tensorflow as tf
from pathlib import Path
from tqdm import tqdm
from PIL import Image as PILImage
from itertools import cycle, islice, chain
from einops import rearrange, repeat
from collections import defaultdict
from tabulate import tabulate
from termcolor import colored

import multiprocessing as mp
import torch.utils.data as data
import torch.nn.functional as F
from torchvision.transforms import (
    InterpolationMode, Compose, Resize, CenterCrop, ToTensor, Normalize
)

from .audio import (
    make_transform, _extract_kaldi_spectrogram 
)
from .ast import make_image_transform
from clip import tokenize

class AudioCapDatasetSrc(data.Dataset):
    """ `__getitem__' loads raw file from disk.
    """
    def __init__(self, cfg, data_name, train, label_map):
        data_path = f"{cfg.data_root}/{data_name}.csv"
        assert os.path.isfile(data_path), f"{data_path} is not a file."
        self.label_map = label_map
        self.num_label = len(label_map)
        label_counts = np.zeros(self.num_label) 
        self.dataset = list()
        with open(data_path, "r") as fr:
            for iline, line in enumerate(fr):
                record = json.loads(line)
                record["captions_bpe"] = tokenize(
                    record["captions"], as_list=True
                ) # add bpe captions
                self.dataset.append(record) 
                if not train and iline + 1 == cfg.eval_samples:
                    break
        self.length = len(self.dataset)
        self.audio_norms = cfg.audio.norms
        self.train = train
        self.cfg = cfg
        
        acfg = cfg.audio
        self.transform_image = make_image_transform(cfg.resolution)
        self.transform_audio, self.transform_fbank = make_transform(acfg)        
        self.kaldi_params = {
            "htk_compat": True, 
            "use_energy": False,
            "window_type": 'hanning', 
            "num_mel_bins": acfg.num_mel_bins, 
            "dither": 0.0, 
            "frame_shift": 10
        }

    def _shuffle(self):
        pass

    def _process_item(self, index):
        akey = "aclip"
        fkey = "frame"
        sub_dir = self.dataset[index]["dir"] 
        name = self.dataset[index]["id"] 
        aclip = self.dataset[index][akey][0] 
        frame = images = self.dataset[index][fkey]
        categories = self.dataset[index]["labels"]
        
        sub_dir = "" if len(sub_dir) == 0 else f"{sub_dir}/"
        aclip_file = f"{self.cfg.data_root}/{sub_dir}{akey}/{name}.{aclip}"
        frame_file = None 
        if self.cfg.imagine:
            assert len(images) != 0, f"no frame exist: |images| = {len(images)}"
            if self.train:
                idx = np.random.choice(len(images), 1)[0]
                ict = np.random.choice(len(categories), 1)[0]
            else:
                idx = int(np.ceil(len(images) / 2)) - 1
                ict = 0 # 1st label
            frame_file = f"{self.cfg.data_root}/{sub_dir}{fkey}/{name}.{images[idx]}"
        
        label = [0]
        captions_bpe = self.dataset[index]["captions_bpe"]
        if self.train:
            icp = np.random.choice(len(captions_bpe), 1)[0]
            text_bpe = captions_bpe[icp]
        else:
            text_bpe = captions_bpe

        item = {"text": text_bpe, "name": name}
        return item, label, aclip_file, frame_file

    def _img2numpy(self, fname):
        if fname is not None: 
            try:
                image = PILImage.open(fname)
            except Exception as e:
                h = w = self.cfg.resolution
                image = PILImage.fromarray(
                    (np.random.rand(h, w, 3) * 256).astype(np.uint8)
                )
                warnings.warn(f"use random image instead because `{e}`.")
            image = self.transform_image(image).cpu().numpy()
        else:
            image = np.array([[[1]]]) 
        return image

    def _wav2fbank(self, index):
        item, label, aclip_file, frame_file = self._process_item(index)
        image = self._img2numpy(frame_file) 
        wf, sr = torchaudio.load(aclip_file)
        wf = wf[:1] #wf.mean(0, keepdim=True)
        wf = wf - wf.mean()

        sampler = np.random if self.cfg.np_rnd else random 

        #if self.train and sampler.random() < self.cfg.mixup_rate:
        if not self.cfg.audio.eval_norms and self.train and sampler.random() < self.cfg.mixup_rate:
            idx_mix = sampler.randint(0, self.length if self.cfg.np_rnd else self.length - 1)
            item_mix, label_mix, aclip_file, _ = self._process_item(idx_mix)
            wf_mix, _ = torchaudio.load(aclip_file)
            wf_mix = wf_mix[:1] #wf_mix.mean(0, keepdim=True)
            wf_mix = wf_mix - wf_mix.mean()
            
            wf_len = wf.shape[1]
            wf_mix = wf_mix[:, :wf_len]
            npad = wf_len - wf_mix.shape[1]
            if npad > 0:
                wf_mix = F.pad(wf_mix, (0, npad), mode='constant', value=0.)

            lambd = np.random.beta(10, 10) # sample lambda from beta distribtion
            wf_mixed = lambd * wf + (1 - lambd) * wf_mix
            wf_mixed = wf_mixed - wf_mixed.mean()
            wf = wf_mixed
            
            label = lambd * np.array(label) + (1 - lambd) * np.array(label_mix)
            label = label.tolist()
        
        audio = torchaudio.compliance.kaldi.fbank(
            wf, 
            sample_frequency=sr, 
            **self.kaldi_params
        )
        
        max_audio_len = self.cfg.max_audio_len
        audio = audio[:max_audio_len]
        npad = max_audio_len - audio.shape[0]
        if npad > 0:
            audio = F.pad(audio, (0, 0, 0, npad), mode='constant', value=0.)
        return item, audio, image, label 

    def __getitem__(self, index):
        item, audio, image, label = self._wav2fbank(index)

        #if self.train and self.transform_fbank is not None:
        if not self.cfg.audio.eval_norms and self.train and self.transform_fbank is not None:
            audio = self.transform_fbank(audio)

        if not self.cfg.audio.eval_norms and len(self.audio_norms) == 2:
            mean, std = self.audio_norms
            audio = (audio - mean) / (std * 2) 

        image = image[None]
        audio = audio[None]
        item.update({"image": image, "audio": audio, "label": label})
        return item 

    def __len__(self):
        return self.length
