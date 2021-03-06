import os
import re
import glob
import json
import torch
import itertools
import torchaudio
import numpy as np
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
from .audioset_cls import print_label_dist, AudiosetNpz, AudiosetSrc 
from clip import tokenize

class AudiosetDatasetNpz(data.Dataset):
    """ `__getitem__' loads .npz from disk.
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

        self.transform_audio, self.transform_fbank = make_transform(cfg.audio)

    def _shuffle(self):
        pass

    def __getitem__(self, index):
        name = self.dataset[index]["id"] 
        aclip = self.dataset[index]["aclip"] 
        frame = self.dataset[index]["frame"]
        categories = self.dataset[index]["labels"]

        aclip_file = f"{self.cfg.data_root}/{aclip}"
        frame_file = f"{self.cfg.data_root}/{frame}"

        images = np.load(frame_file)
        images = [images[key] for key in images.files if len(images[key]) != 0]
        assert len(images) != 0, f"no frame exist: |images| = {len(images)}"
        if self.train:
            idx = np.random.choice(len(images), 1)[0]
            ict = np.random.choice(len(categories), 1)[0]
        else:
            idx = int(np.ceil(len(images) / 2)) - 1
            ict = 0 # 1st label
        image = images[idx]

        max_audio_len = self.cfg.max_audio_len
        audio = np.load(aclip_file)["flag"] # (..., time, freq): `flag' is used as the key accidentally

        if self.cfg.audio.normalized: # normalize along feature dim
            audio /= np.max(np.abs(audio), axis=1)[:, None]

        if not self.cfg.audio.eval_norms and len(self.audio_norms) == 2:
            mean, std = self.audio_norms
            audio = (audio - mean) / std 

        if not self.cfg.audio.eval_norms and self.train and self.transform_fbank is not None:
            audio = self.transform_fbank(audio)

        npad = max_audio_len - audio.shape[0]
        if npad > 0:
            audio = np.pad(audio, ((0, npad), (0, 0)), "constant", constant_values=(0., 0.))

        image = image[None]
        audio = audio[None]
        
        if not self.cfg.clf: 
            category = categories[ict]
            label, _, text_int = self.label_map[category] 
        else: # classification task
            label_set = set([self.label_map[category][0] for category in categories])
            label = [1 if i in label_set else 0 for i in range(self.num_label)] 
            text_int = [0] # TODO concatenate all text pieces

        item = {"image": image, "audio": audio, "text": text_int, "label": label, "name": name}
        return item 

    def __len__(self):
        return self.length

class ImageAudioCollator:
    def __init__(self, device=torch.device("cpu")):
        # RuntimeError: cannot pin 'torch.cuda.FloatTensor' only dense CPU tensors can be pinned
        # when pin_memory is true, the collator has to return CPU tensors
        self.device = device

    def __call__(self, records):
        union = { 
            k: [record.get(k) for record in records] for k in set().union(*records) 
        } 
        name = union["name"] 
        label = union["label"]
        text_list = union["text"]
        if isinstance(text_list[0][0], np.ndarray): # pre-computed text embeddings
            text = np.concatenate(text_list, axis=0) # (1, H) -> (b, H)
        else:
            if isinstance(text_list[0][0], int): # train / test clf
                label = np.array(label)
            elif isinstance(text_list[0][0], list): # test retrieval
                text_list = list(itertools.chain.from_iterable(text_list))
                #name = list(itertools.chain.from_iterable(name))
                #label = # list of label lists
            else:
                raise ValueError(f"unrecognized `{type(text_list[0][0])}`")
            # https://stackoverflow.com/a/38619333
            text = np.array(list(itertools.zip_longest(*text_list, fillvalue=0))).T
        return (
            np.concatenate(union["image"], axis=0), 
            np.concatenate(union["audio"], axis=0),
            text, label, name,
        )

def build_audioset_clf_dataloader(cfg, data_name, label_map, shuffle=True, train=True, filters=None):
    ddp_mode = torch.distributed.is_initialized()
    rcfg = cfg.running
    weighted = train and rcfg.weighted_sampling
    if data_name.startswith("src"):
        if not rcfg.force_npz:
            dataset = AudiosetSrc(rcfg, data_name, train, label_map, weighted, filters)
        else:
            dataset = AudiosetNpz(rcfg, data_name, train, label_map, weighted)
    elif data_name.startswith("npz"):
        dataset = AudiosetDatasetNpz(rcfg, data_name, train, label_map, weighted)
    else:
        dataset = AudiosetSrc(rcfg, data_name, train, label_map, weighted)
        #raise ValueError(f"unrecognized data file `{data_name}`.")
    if ddp_mode:
        assert cfg.optimizer.batch_size % cfg.num_gpus == 0
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, shuffle=shuffle
        ) 
        per_device_batch_size = cfg.optimizer.batch_size // cfg.num_gpus
    else:
        if not weighted:
            sampler = (
                torch.utils.data.RandomSampler(dataset) if shuffle else
                torch.utils.data.SequentialSampler(dataset)
            )
        else:
            sampler = torch.utils.data.WeightedRandomSampler(
                dataset.sample_weights, len(dataset), replacement=True
            )
        per_device_batch_size = cfg.optimizer.batch_size
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=per_device_batch_size,
        collate_fn=ImageAudioCollator(),
        num_workers=(0 if ddp_mode else cfg.num_proc),
        pin_memory=True,
        sampler=sampler,
        drop_last=(True if ddp_mode else False),
    )
    return sampler, dataloader
