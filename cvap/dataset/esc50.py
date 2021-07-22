import os
import copy
import glob
import json
import torch
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

from torchvision.transforms import Compose, ToTensor

def _extract_kaldi_spectrogram(filename, params, max_audio_len=1000):
    waveform, sample_rate = torchaudio.load(f"{filename}")
    fbank_feat = torchaudio.compliance.kaldi.fbank(
        waveform,
        sample_frequency=sample_rate,
        **params,
    )
    fbank_feat = fbank_feat[:max_audio_len]
    return fbank_feat.numpy()

class ImageAudioDatasetSrc(data.Dataset):
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

        self.transform = Compose([
            lambda x: x.T,
            ToTensor(), # will add a new dim 0 
            torchaudio.transforms.FrequencyMasking(cfg.freq_mask_param),
            torchaudio.transforms.TimeMasking(cfg.time_mask_param),
            lambda x: x.squeeze(0).T,
        ]) # return Tensor or .numpy() if numpy.array is wanted
        
        self.kaldi_params = {
            "use_log_fbank": cfg.use_log_fbank,
            "frame_length": cfg.frame_length,
            "frame_shift": cfg.frame_shift,
            "window_type": cfg.window_type,
            "num_mel_bins": cfg.num_mel_bins,
            "high_freq": cfg.high_freq,
            "low_freq": cfg.low_freq,
        }

    def _shuffle(self):
        pass

    def __getitem__(self, index):
        label_str = self.dataset[index]["label_str"] 
        label_int = self.dataset[index]["label_int"] 
        aclip = self.dataset[index]["aclip"] 

        aclip_file = f"{self.cfg.data_root}/{aclip}"

        max_audio_len = self.cfg.max_audio_len
        audio = _extract_kaldi_spectrogram(
            aclip_file, self.kaldi_params, max_audio_len=max_audio_len
        ) # (..., time, freq)
        
        if self.train and (self.cfg.freq_mask_param > 1 or self.cfg.time_mask_param > 1): # data augmentation
            audio = self.transform(audio)

        npad =  self.cfg.max_audio_len - audio.shape[0]
        if npad > 0:
            audio = np.pad(audio, ((0, npad), (0, 0)), "constant", constant_values=(0., 0.))
        
        audio = audio[None]

        item = {"audio": audio, "label_int": label_int, "label_str": label_str}
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
        return (
            np.concatenate(union["audio"], axis=0),
            np.array(union["label_int"]),
            union["label_str"],
        )

def build_dataloader(cfg, data_list, shuffle=True, train=True):
    ddp_mode = torch.distributed.is_initialized()
    rcfg = cfg.running
    dataset = ImageAudioDatasetSrc(rcfg, data_list, train)
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
        collate_fn=ImageAudioCollator(),
        num_workers=(0 if ddp_mode else cfg.num_proc),
        pin_memory=True,
        sampler=sampler,
        drop_last=(True if ddp_mode else False),
    )
    return sampler, dataloader

def build_dataloader_list_esc50(cfg):
    rcfg = cfg.running
    data_path = f"{rcfg.data_root}/meta/{rcfg.data_name}.csv"
    assert os.path.isfile(data_path), f"{data_path} is not a file."
    meta = np.loadtxt(data_path, delimiter=",", dtype="str", skiprows=1)
    nfold = 5
    folds = [[] for _ in range(nfold)] 
    lid2str = dict()
    for i, row in enumerate(meta):
        filename, fold, target, category, _, _, _ = row 
        item = {
            "aclip": f"audio/{filename}", 
            "label_int": int(target), 
            "label_str": category
        } 
        folds[int(fold) - 1].append(item)
        lid2str[target] = category
    
    loader_tuple = tuple()
    for i in range(nfold):
        train_list = []
        for j in range(nfold): 
            if j == i:
                continue
            train_list.extend(copy.deepcopy(folds[j]))
        eval_list = copy.deepcopy(folds[i])
        # print(len(train_list), len(eval_list), len(train_list) + len(eval_list))
        # lazy loading 
        loader_tuple += ((
            lambda data_list=train_list: build_dataloader(cfg, data_list),
            lambda data_list=eval_list: build_dataloader(cfg, data_list, shuffle=False, train=False)
        ),)
    return loader_tuple, lid2str 

def build_dataloader_list_us8k(cfg):
    rcfg = cfg.running
    data_path = f"{rcfg.data_root}/metadata/{rcfg.data_name}.csv"
    assert os.path.isfile(data_path), f"{data_path} is not a file."
    meta = np.loadtxt(data_path, delimiter=",", dtype="str", skiprows=1)
    nfold = 10 
    folds = [[] for _ in range(nfold)] 
    lid2str = dict()
    for i, row in enumerate(meta):
        filename, _, _, _, _, fold, target, category = row
        item = {
            "aclip": f"audio/fold{fold}/{filename}", 
            "label_int": int(target), 
            "label_str": category
        } 
        folds[int(fold) - 1].append(item)
        lid2str[target] = category
    
    loader_tuple = tuple()
    for i in range(nfold):
        train_list = []
        for j in range(nfold): 
            if j == i:
                continue
            train_list.extend(copy.deepcopy(folds[j]))
        eval_list = copy.deepcopy(folds[i])
        # print(len(train_list), len(eval_list), len(train_list) + len(eval_list))
        # lazy loading 
        loader_tuple += ((
            lambda data_list=train_list: build_dataloader(cfg, data_list),
            lambda data_list=eval_list: build_dataloader(cfg, data_list, shuffle=False, train=False)
        ),)
    return loader_tuple, lid2str 

def build_dataloader_list(cfg):
    if cfg.running.data_name == "esc50":
        return build_dataloader_list_esc50(cfg)
    elif cfg.running.data_name == "UrbanSound8K":
        return build_dataloader_list_us8k(cfg)
    else:
        raise ValueError(f"unrecognized dataset `{cfg.running.data_name}`.") 

