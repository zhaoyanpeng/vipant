import os
import glob
import json
import torch
import itertools
import torchaudio
import numpy as np
import tensorflow as tf
from pathlib import Path
from tqdm import tqdm
from itertools import cycle, islice, chain
from einops import rearrange, repeat
from collections import defaultdict

import multiprocessing as mp
import torch.utils.data as data
import torch.nn.functional as F

from .audio import (
    make_transform, _extract_kaldi_spectrogram 
)
from .ast import ASTSrc
from .audioset_cap import AudioCapDatasetSrc
from .audioset_ast import AudiosetDatasetNpz, ImageAudioCollator
from clip import tokenize

###
# this file ocntains the very first implementations of AudioSet data loader.
# we now have clf-focused loader in ast.py and audioset_ast.py
# and contrastive-focused loader in image_audio.py and this file.
###

def collect_ytid(csv_root, csv_list):
    ids = defaultdict(list)
    nrow = 0
    for fname in csv_list:
        ifile = f"{csv_root}/{fname}.csv"
        with open(ifile, "r") as fr:
            for _ in range(3):
                next(fr)
            for irow, row in enumerate(fr):
                row = row.split(", ")
                ids[row[0].strip()].append(
                    (row[1].strip(), row[2].strip(), row[3].strip('" \n').split(","))
                )
                nrow += 1
    return list(ids.keys()), ids

def build_audioset_label_map(data_root, label_map="ontology,eval_segments", prompt=""): 
    file_list = label_map.split(",")
    ontology, label_files = file_list[0], file_list[1:]
    label_path = f"{data_root}/{ontology}.json"
    label_real = f"{data_root}/{label_files[0]}.csv"
    assert os.path.isfile(label_path) and os.path.isfile(label_real), (
        "please specify a valid `ontology` file (ontology) and `eval` file (eval_segments)."
    )
    category_list = list()
    ontology = json.load(open(label_path, "r"))
    for item in ontology:
        category = item["id"]
        category_list.append(
            (category, prompt + " " + item["name"].lower())
        )
    text_list = [item[1] for item in category_list]
    label_int = tokenize(text_list, as_list=True)
    #label_map = {category_list[i][0]: (i, category_list[i][1], label_int[i]) for i in range(len(category_list))}

    _, ytid_dict = collect_ytid(data_root, label_files)

    label_set = set(itertools.chain.from_iterable(
        v[0][2] for _, v in ytid_dict.items()
    ))
    category_list = [item for item in category_list if item[0] in label_set]
    label_map = {category_list[i][0]: (i, category_list[i][1], label_int[i]) for i in range(len(category_list))}
    #print(text_list, len(label_set))
    #print(label_map, len(label_map))
    return label_map 

def build_audioset_dataloader(cfg, data_name, label_map, shuffle=True, train=True):
    ddp_mode = torch.distributed.is_initialized()
    rcfg = cfg.running
    if data_name.startswith("src"):
        if not rcfg.force_npz:
            dataset = ASTSrc(rcfg, data_name, train, label_map, False)
        else:
            dataset = ASTNpz(rcfg, data_name, train, label_map, False)
    elif data_name.startswith("npz"):
        dataset = AudiosetDatasetNpz(rcfg, data_name, train, label_map, False)
    elif data_name.startswith("audiocaps"): # audio captioning
        dataset = AudioCapDatasetSrc(rcfg, data_name, train, label_map)
    else:
        raise ValueError(f"unrecognized data file `{data_name}`.")
    if ddp_mode:
        assert cfg.optimizer.batch_size % cfg.num_gpus == 0
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, shuffle=shuffle
        ) 
        per_device_batch_size = cfg.optimizer.batch_size // cfg.num_gpus
    else:
        sampler = (
            torch.utils.data.RandomSampler(dataset) if shuffle else
            torch.utils.data.SequentialSampler(dataset)
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
