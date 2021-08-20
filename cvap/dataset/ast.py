import os
import re
import glob
import json
import torch
import random
import itertools
import torchaudio
import numpy as np
import tensorflow as tf
from pathlib import Path
from tqdm import tqdm
from itertools import cycle, islice, chain
from einops import rearrange, repeat
from collections import defaultdict
from tabulate import tabulate
from termcolor import colored

import multiprocessing as mp
import torch.utils.data as data
import torch.nn.functional as F

from .audio import (
    make_transform, _extract_kaldi_spectrogram 
)
from clip import tokenize

def print_label_dist(cfg, echo, label_counts, label_map, ncol=30):
    def short_name(x):
        if len(x) > 15:
            return x[:13] + ".."
        return x
    data = list(itertools.chain(*[
        [short_name(label_map[i]), int(v)] for i, v in enumerate(label_counts)
    ]))
    total_num_instances = sum(data[1::2])
    data.extend([None] * (ncol - (len(data) % ncol)))
    data = itertools.zip_longest(*[data[i::ncol] for i in range(ncol)])
    table = tabulate(
        data,
        headers=["category", "#"] * (ncol // 2),
        tablefmt="pipe",
        numalign="right",
        stralign="center",
    )
    msg = colored(table, "cyan")
    echo(f"Distribution of instances among all {len(label_map)} categories:\n{msg}")

class ASTSrc(data.Dataset):
    """ `__getitem__' loads raw file from disk.
    """
    def __init__(self, cfg, data_name, train, label_map, weighted):
        data_path = f"{cfg.data_root}/{data_name}.csv"
        assert os.path.isfile(data_path), f"{data_path} is not a file."
        self.label_map = label_map
        self.num_label = len(label_map)
        label_counts = np.zeros(self.num_label) 
        self.dataset = list()
        with open(data_path, "r") as fr:
            for iline, line in enumerate(fr):
                record = json.loads(line)
                self.dataset.append(record) 
                if not train and iline + 1 == cfg.eval_samples:
                    break
                if weighted: # save label distribution
                    for category in record["labels"]:
                        label_counts[
                            self.label_map[category][0]
                        ] += 1
        self.length = len(self.dataset)
        if weighted: # compute sample weight
            lid2label = {v[0]: re.sub(f"^{cfg.prompt}", "", v[1]).strip() for _, v in label_map.items()}
            print_label_dist(cfg, print, label_counts, lid2label, ncol=18)
            self.sample_weights = np.zeros(self.length)
            label_counts = 1000.0 / (label_counts + 1.)
            for i, record in enumerate(self.dataset):
                for category in record["labels"]:
                    self.sample_weights[i] += label_counts[
                        self.label_map[category][0]
                    ]
        self.audio_norms = cfg.audio.norms
        self.train = train
        self.cfg = cfg
        
        acfg = cfg.audio
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
        akey = "clip"
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

        if not self.cfg.clf: 
            category = categories[ict]
            label, _, text_int = self.label_map[category] 
        else: # classification task
            label_set = set([self.label_map[category][0] for category in categories])
            label = [1 if i in label_set else 0 for i in range(self.num_label)] 
            text_int = [0] # TODO concatenate all text pieces

        item = {"text": text_int, "name": name}
        return item, label, aclip_file, frame_file

    def _img2numpy(self, fname):
        if fname is not None: # TODO
            image = np.array([[[1]]]) 
        else:
            image = np.array([[[1]]]) 
        return image

    def _wav2fbank(self, index):
        item, label, aclip_file, frame_file = self._process_item(index)
        image = self._img2numpy(frame_file) 
        wf, sr = torchaudio.load(aclip_file)
        wf = wf[:1] #wf.mean(0, keepdim=True)
        wf = wf - wf.mean()

        if self.train and np.random.random() < self.cfg.mixup_rate:
            idx_mix = np.random.randint(self.length)
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

        #if not self.cfg.audio.eval_norms and 
        if self.train and self.transform_fbank is not None:
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
