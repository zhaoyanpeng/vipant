import os
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

from . import PairImageSpectrogramTFRecords
from .audio import (
    make_transform, _extract_kaldi_spectrogram
)

class ImageAudioDataset(data.Dataset):
    def __init__(self, cfg, data_name, train):
        data_path = f"{cfg.data_root}/{data_name}"
        records = PairImageSpectrogramTFRecords(
            data_path, 1, max_audio_len=cfg.max_audio_len
        )
        self.dataset = list()
        for iline, record in enumerate(records):
            self.dataset.append(record) 
            if not train and iline + 1 == cfg.eval_samples:
                break
        self.length = len(self.dataset)

    def _shuffle(self):
        pass

    def __getitem__(self, index):
        return self.dataset[index] 

    def __len__(self):
        return self.length

class ImageAudioDatasetNpz(data.Dataset):
    """ `__getitem__' loads .npz from disk.
    """
    def __init__(self, cfg, data_name, train):
        data_path = f"{cfg.data_root}/{data_name}.csv"
        assert os.path.isfile(data_path), f"{data_path} is not a file."
        self.dataset = list()
        with open(data_path, "r") as fr:
            for iline, line in enumerate(fr):
                record = json.loads(line)
                self.dataset.append(record) 
                if not train and iline + 1 == cfg.eval_samples:
                    break
        if train and cfg.train_samples > 0. and cfg.train_samples < 1.:
            k = int(len(self.dataset) * cfg.train_samples)
            self.dataset = np.random.choice(self.dataset, k, replace=False) 
        self.length = len(self.dataset)
        self.train = train
        self.cfg = cfg

        self.transform_audio, self.transform_fbank = make_transform(cfg.audio)

    def _shuffle(self):
        pass

    def __getitem__(self, index):
        name = self.dataset[index]["id"] 
        aclip = self.dataset[index]["aclip"] 
        frame = self.dataset[index]["frame"]

        aclip_file = f"{self.cfg.data_root}/{aclip}"
        frame_file = f"{self.cfg.data_root}/{frame}"

        images = np.load(frame_file)
        images = [images[key] for key in images.files if len(images[key]) != 0]
        assert len(images) != 0, f"no frame exist: |images| = {len(images)}"
        if self.train:
            idx = np.random.choice(len(images), 1)[0]
        else:
            idx = int(np.ceil(len(images) / 2)) - 1
        image = images[idx] 

        max_audio_len = self.cfg.max_audio_len
        audio = np.load(aclip_file)["flag"] # (..., time, freq): `flag' is used as the key accidentally

        if self.train and self.transform_fbank is not None:
            audio = self.transform_fbank(audio)

        npad =  max_audio_len - audio.shape[0]
        if npad > 0:
            audio = np.pad(audio, ((0, npad), (0, 0)), "constant", constant_values=(0., 0.))
        
        image = image[None]
        audio = audio[None]

        item = {"image": image, "audio": audio, "name": name}
        return item 

    def __len__(self):
        return self.length

class ImageAudioDatasetSrc(data.Dataset):
    """ `__getitem__' loads raw file from disk.
    """
    def __init__(self, cfg, data_name, train):
        data_path = f"{cfg.data_root}/{data_name}.csv"
        assert os.path.isfile(data_path), f"{data_path} is not a file."
        self.dataset = list()
        with open(data_path, "r") as fr:
            for iline, line in enumerate(fr):
                record = json.loads(line)
                self.dataset.append(record) 
                if not train and iline + 1 == cfg.eval_samples:
                    break
        if train and cfg.train_samples > 0. and cfg.train_samples < 1.:
            k = int(len(self.dataset) * cfg.train_samples)
            self.dataset = np.random.choice(self.dataset, k, replace=False) 
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
        akey = "aclip"
        fkey = "frame_224"
        dir = self.dataset[index]["dir"] 
        name = self.dataset[index]["id"] 
        aclip = self.dataset[index][akey][0] 
        frame = self.dataset[index][fkey]

        aclip_file = f"{self.cfg.data_root}/{dir}/{akey}/{name}.{aclip}"
        frame_file = f"{self.cfg.data_root}/{dir}/{fkey}/{name}.{frame}"

        max_audio_len = self.cfg.max_audio_len

        images = np.load(frame_file)
        images = [images[key] for key in images.files if len(images[key]) != 0]
        assert len(images) != 0, f"no frame exist: |images| = {len(images)}"
        if self.train:
            idx = np.random.choice(len(images), 1)[0]
        else:
            idx = int(np.ceil(len(images) / 2)) - 1
        image = images[idx] 
        
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
        if npad > 0:
            audio = np.pad(audio, ((0, npad), (0, 0)), "constant", constant_values=(0., 0.))
        
        image = image[None]
        audio = audio[None]

        item = {"image": image, "audio": audio, "name": name}
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
            np.concatenate(union["image"], axis=0), 
            np.concatenate(union["audio"], axis=0),
            union["name"],
        )
        union = {
            "image": torch.tensor(
                np.concatenate(union["image"], axis=0), device=self.device
            ), 
            "audio": torch.tensor(
                np.concatenate(union["audio"], axis=0), device=self.device
            ).unsqueeze(1), 
            "name": union["name"],
        }
        return union

def build_dataloader(cfg, data_name, shuffle=True, train=True):
    ddp_mode = torch.distributed.is_initialized()
    rcfg = cfg.running
    if data_name.startswith("src"):
        dataset = ImageAudioDatasetSrc(rcfg, data_name, train)
    elif data_name.startswith("npz"):
        dataset = ImageAudioDatasetNpz(rcfg, data_name, train)
    else:
        dataset = ImageAudioDataset(rcfg, data_name, train)
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
