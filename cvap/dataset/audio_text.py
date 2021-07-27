import os
import csv
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

import multiprocessing as mp
import torch.utils.data as data
import torch.nn.functional as F

from .audio import (
    make_transform, build_dataloader_list, _extract_kaldi_spectrogram 
)
from clip import tokenize

class AudioTextDatasetSrc(data.Dataset):
    """ `__getitem__' loads raw file from disk.
    """
    def __init__(self, cfg, data_list, train):
        self.dataset = list()
        for iline, record in enumerate(data_list):
            self.dataset.append(record) 
            if not train and iline + 1 == cfg.eval_samples:
                break
        self.length = len(self.dataset)
        self.train = train
        self.cfg = cfg
        
        acfg = cfg.audio
        self.transform_audio, self.transform_fbank = make_transform(acfg)        
        self.kaldi_params = {
            "use_log_fbank": acfg.use_log_fbank,
            "frame_length": acfg.frame_length,
            "frame_shift": acfg.frame_shift,
            "window_type": acfg.window_type,
            "num_mel_bins": acfg.num_mel_bins,
            "high_freq": acfg.high_freq,
            "low_freq": acfg.low_freq,
        }

    def _shuffle(self):
        pass

    def __getitem__(self, index):
        label_str = self.dataset[index]["label_str"] 
        label_int = self.dataset[index]["label_int_bpe"] 
        aclip = self.dataset[index]["aclip"] 

        aclip_file = f"{self.cfg.data_root}/{aclip}"

        max_audio_len = self.cfg.max_audio_len
        audio = _extract_kaldi_spectrogram(
            aclip_file, 
            self.kaldi_params, 
            train=self.train, 
            max_audio_len=max_audio_len,
            transform_audio=(self.transform_audio if self.train else None)
        ) # (..., time, freq)
        
        if self.train and self.transform_fbank is not None:
            audio = self.transform_fbank(audio)

        npad =  self.cfg.max_audio_len - audio.shape[0]
        if npad > 0: # always pad to the right
            audio = np.pad(audio, ((0, npad), (0, 0)), "constant", constant_values=(0., 0.))
        
        audio = audio[None]

        if self.train:
            idx = np.random.choice(len(label_int), 1)[0]
            text = label_int[idx]
            name = label_str[idx] 
        else:
            text = label_int
            name = label_str

        item = {"audio": audio, "text": text, "name": name}
        return item 

    def __len__(self):
        return self.length

class AudioTextCollator:
    def __init__(self, device=torch.device("cpu")):
        # RuntimeError: cannot pin 'torch.cuda.FloatTensor' only dense CPU tensors can be pinned
        # when pin_memory is true, the collator has to return CPU tensors
        self.device = device

    def __call__(self, records):
        union = { 
            k: [record.get(k) for record in records] for k in set().union(*records) 
        } 
        name = union["name"] 
        text_list = union["text"]
        if isinstance(text_list[0][0], int): # train
            pass 
            """ https://stackoverflow.com/a/43149308
            lengths = [len(x) for x in text_list]
            max_len = max(lengths)
            text = np.zeros((len(text_list), max_len), int)
            mask = np.arange(max_len) < np.array(lengths)[:, None]
            text[mask] = np.concatenate(text_list)
            """
        elif isinstance(text_list[0][0], list): # test
            text_list = list(itertools.chain.from_iterable(text_list))
            name = list(itertools.chain.from_iterable(name))
        else:
            raise ValueError(f"unrecognized `{type(text_list[0][0])}`")
        text = np.array(list(itertools.zip_longest(*text_list, fillvalue=0))).T
        return (
            np.concatenate(union["audio"], axis=0),
            text,
            name,
        )

def build_dataloader(cfg, data_list, dataset_cls, shuffle=True, train=True):
    ddp_mode = torch.distributed.is_initialized()
    rcfg = cfg.running
    if isinstance(dataset_cls, str):
        dataset = eval(dataset_cls)(rcfg, data_list, train)
    else:
        dataset = dataset_cls(rcfg, data_list, train)
    if ddp_mode:
        assert self.cfg.optimizer.batch_size % self.cfg.num_gpus == 0
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
        collate_fn=AudioTextCollator(),
        num_workers=(0 if ddp_mode else cfg.num_proc),
        pin_memory=True,
        sampler=sampler,
        drop_last=(True if ddp_mode else False),
    )
    return sampler, dataloader

def build_dataloader_clotho(cfg, data_name, shuffle=True, train=True):
    rcfg = cfg.running
    fold = data_name.rsplit("_", 1)[-1] # {development, validation, evaluation} 
    data_path = f"{rcfg.data_root}/{data_name}.csv"
    assert os.path.isfile(data_path), f"{data_path} is not a file."
    dataset = list()
    with open(data_path, "r") as fr:
        meta = csv.DictReader(fr)
        for i, row in enumerate(meta):
            filename = row["file_name"]
            captions = [row[f"caption_{icap}"] for icap in range(1, 6)]
            label_int_bpe = tokenize(captions, as_list=True)
            item = {
                "aclip": f"{fold}/{filename}", 
                "label_int_bpe": label_int_bpe,
                "label_int_w2v": [], 
                "label_str": captions 
            } 
            dataset.append(item)
            if i > 10:
                pass #break
        #print(dataset)
    return build_dataloader(cfg, dataset, AudioTextDatasetSrc, shuffle=shuffle, train=train)

def build_audio_text_dataloader(cfg, data_name, *args, shuffle=True, train=True, **kwargs):
    if data_name.startswith("clotho"):
        return build_dataloader_clotho(
            cfg, data_name, shuffle=shuffle, train=train
        )
    else:
        raise ValueError(f"unrecognized dataset `{data_name}`.") 
