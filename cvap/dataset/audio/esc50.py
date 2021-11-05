import os
import re
import copy
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

from clip import tokenize
from .transform import make_transform, RandomCrop

from mreserve.preprocess import video_to_segments, preprocess_video

def _extract_kaldi_spectrogram(
    filename, params, train=True, mean_channel=False, zero_mean_wf=False, max_audio_len=1000, transform_audio=None
):
    waveform, sample_rate = torchaudio.load(filename)
    if mean_channel: # mean along channel # TODO else branch should take a specific channel
        waveform = waveform.mean(0, keepdim=True)
    if transform_audio is not None:
        waveform = transform_audio(waveform) 
    waveform = RandomCrop.random_crop(
        waveform, int((max_audio_len / 100 + 0.05) * sample_rate), train=train
    ) # divided by 100 because kaldi has a frame shift of 10, additional 0.05s
    if zero_mean_wf: # TODO should extract the 1st channel before the mean
        waveform = waveform - waveform.mean()
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
        self.audio_norms = cfg.audio.norms
        self.length = len(self.dataset)
        self.train = train
        self.cfg = cfg
        
        acfg = cfg.audio
        self.transform_audio, self.transform_fbank = make_transform(acfg)        
        if not self.cfg.audio.zero_mean_wf:
            self.kaldi_params = {
                "use_log_fbank": acfg.use_log_fbank,
                "frame_length": acfg.frame_length,
                "frame_shift": acfg.frame_shift,
                "window_type": acfg.window_type,
                "num_mel_bins": acfg.num_mel_bins,
                "high_freq": acfg.high_freq,
                "low_freq": acfg.low_freq,
            } # old configs
        else:
            self.kaldi_params = {
                "htk_compat": True,
                "use_energy": False,
                "window_type": 'hanning',
                "num_mel_bins": acfg.num_mel_bins,
                "dither": 0.0,
                "frame_shift": 10
            } # new configs

    def _shuffle(self):
        pass

    def __getitem__(self, index):
        label_str = self.dataset[index]["label_str"] 
        label_int = self.dataset[index]["label_int"] 
        aclip = self.dataset[index]["aclip"] 

        aclip_file = f"{self.cfg.data_root}/{aclip}"

        max_audio_len = self.cfg.max_audio_len
        audio = _extract_kaldi_spectrogram(
            aclip_file,
            self.kaldi_params,
            train=self.train,
            max_audio_len=max_audio_len,
            zero_mean_wf=self.cfg.audio.zero_mean_wf,
            transform_audio=(
                self.transform_audio if self.train and not self.cfg.audio.eval_norms else None
            )
        ) # (..., time, freq)
        
        npad =  self.cfg.max_audio_len - audio.shape[0]
        if npad > 0: # always pad to the right
            audio = np.pad(audio, ((0, npad), (0, 0)), "constant", constant_values=(0., 0.))

        if not self.cfg.audio.eval_norms and len(self.audio_norms) == 2:
            mean, std = self.audio_norms
            audio = (audio - mean) / std
        
        #if self.train and self.transform_fbank is not None:
        if not self.cfg.audio.eval_norms and self.train and self.transform_fbank is not None:
            audio = self.transform_fbank(audio)

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

class ImageAudioDataset4Mreserve(data.Dataset):
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

    def _shuffle(self):
        pass

    def __getitem__(self, index):
        label_str = self.dataset[index]["label_str"]
        label_int = self.dataset[index]["label_int"]
        aclip = self.dataset[index]["aclip"]
        audio = np.array([[[1]]])

        aclip_file = f"{self.cfg.data_root}/{aclip}"
        acfg = self.cfg.audio

        video_segments = video_to_segments(
            aclip_file,
            end_trim=acfg.end_trim,
            segment_gap=acfg.segment_gap,
            pad_segment=acfg.pad_segment,
            min_duration=acfg.min_duration,
            time_interval=acfg.time_interval,
        )

        video_segments = video_segments[:8]
        dummy_segment = copy.deepcopy(video_segments[0])
        video_segments.insert(0, dummy_segment)

        video_segments[0]['text'] = f'{self.cfg.text}'
        video_segments[0]['use_text_as_input'] = True
        for seg in video_segments[1:]:
            seg['use_text_as_input'] = False
        assert len(video_segments) >= 2, f"Require at least 2 video segments."

        video_pre = preprocess_video(video_segments, output_grid_size=acfg.grid_size, verbose=acfg.verbose)

        item = {"video": video_pre, "audio": audio, "label_int": label_int, "label_str": label_str}
        return item

    def __len__(self):
        return self.length

class ImageAudioCollator4Mreserve:
    def __call__(self, records):
        union = {
            k: [record.get(k) for record in records] for k in set().union(*records)
        }
        return (
            np.concatenate(union["audio"], axis=0),
            np.array(union["label_int"]),
            union["label_str"],
            union["video"]
        )

def build_dataloader(cfg, data_list, shuffle=True, train=True, mreserve=False):
    ddp_mode = torch.distributed.is_initialized()
    rcfg = cfg.running
    dataset = (
        ImageAudioDatasetSrc(rcfg, data_list, train) if not mreserve
        else ImageAudioDataset4Mreserve(rcfg, data_list, train)
    )
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
        per_device_batch_size = rcfg.batch_size
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=per_device_batch_size,
        collate_fn=(ImageAudioCollator4Mreserve() if mreserve else ImageAudioCollator()),
        num_workers=(0 if ddp_mode else cfg.num_proc),
        pin_memory=True,
        sampler=sampler,
        drop_last=(True if ddp_mode else False),
    )
    return sampler, dataloader

def build_dataloader_list_esc50(cfg, mreserve=False):
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
        lid2str[int(target)] = category
    
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
            lambda data_list=train_list: build_dataloader(cfg, data_list, mreserve=mreserve),
            lambda data_list=eval_list: build_dataloader(cfg, data_list, shuffle=False, train=False, mreserve=mreserve)
        ),)

    label_path = f"{rcfg.data_root}/meta/{rcfg.prompt}.json"
    if not os.path.isfile(label_path):
        prompt = rcfg.prompt.strip()
        prompt = "" if prompt == "" else prompt + " "
        lid2int = [prompt + lid2str[i].replace("_", " ") for i in range(len(lid2str))]
        label_map = {i: i for i in range(len(lid2str))}
    else:
        topk = 4 
        label_map = json.load(open(label_path, "r"))
        text_list = [
            label_map[lid2str[i].replace("_", " ")][:topk] for i in range(len(lid2str))
        ]
        lid2int = list(itertools.chain.from_iterable(text_list))
        lid2int = [re.sub("^a photo of", "the sound of", text) for text in lid2int]
        assert len(lid2int) == len(text_list) * topk, f"unbalanced label mapping: {len(text_list)}x{topk} -> {len(lid2int)}"
        label_map = {i: i // topk for i in range(len(lid2str) * topk)}
    print(lid2int)
    lid2int = tokenize(lid2int, as_list=True)
    lid2int = np.array(list(itertools.zip_longest(*lid2int, fillvalue=0))).T
    return loader_tuple, lid2str, lid2int, label_map

def build_dataloader_list_us8k(cfg, mreserve=False):
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
        lid2str[int(target)] = category
    
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
            lambda data_list=train_list: build_dataloader(cfg, data_list, mreserve=mreserve),
            lambda data_list=eval_list: build_dataloader(cfg, data_list, shuffle=False, train=False, mreserve=mreserve)
        ),)

    label_path = f"{rcfg.data_root}/meta/{rcfg.prompt}.json"
    if not os.path.isfile(label_path):
        prompt = rcfg.prompt.strip()
        prompt = "" if prompt == "" else prompt + " "
        lid2int = [prompt + lid2str[i].replace("_", " ") for i in range(len(lid2str))]
        #lid2int = [lid2str[i].replace("_", " ") + " " + prompt for i in range(len(lid2str))]
        label_map = {i: i for i in range(len(lid2str))}
    else:
        lid2int = [lid2str[i].replace("_", " ") for i in range(len(lid2str))]
        pass
    print(lid2int)
    lid2int = tokenize(lid2int, as_list=True)
    lid2int = np.array(list(itertools.zip_longest(*lid2int, fillvalue=0))).T
    return loader_tuple, lid2str, lid2int, None

def build_dataloader_list_audioset(cfg, mreserve=False):
    rcfg = cfg.running
    data_path = f"{rcfg.data_root}/{rcfg.eval_name}.csv"
    assert os.path.isfile(data_path), f"{data_path} is not a file."
    from .. import build_audioset_label_map as build_label_map
    label_map = build_label_map(rcfg.data_root, label_map=rcfg.label_map, prompt="")
    lid2int = [0] * len(label_map)
    lid2str = [0] * len(label_map)
    for k, v in label_map.items():
        lid2int[v[0]] = v[2]
        lid2str[v[0]] = v[1]
    checksum = []
    for i, (k, v) in enumerate(label_map.items()):
        checksum.append(v[-1] == lid2int[v[0]])
        if i < 600 and rcfg.zero_shot:
            pass #print(k, v)
    #print(all(checksum))

    eval_list = list()
    with open(data_path, "r") as fr:
        for iline, line in enumerate(fr):
            record = json.loads(line)
            name = record["id"]
            sub_dir = record["dir"]
            sub_dir = "" if len(sub_dir) == 0 else f"{sub_dir}/"
            akey = "clip" if "clip" in record else "aclip"
            aclip = record[akey][0]

            label_int_set = set()
            label_str_set = set()
            for category in record["labels"]:
                label_str_set.add(label_map[category][1])
                label_int_set.add(label_map[category][0])

            label_int = [1 if i in label_int_set else 0 for i in range(len(label_map))]
            label_str = "<O>".join(list(label_str_set))

            item = {
                "aclip": f"{sub_dir}{akey}/{name}.{aclip}",
                "label_int": label_int,
                "label_str": label_str,
            }
            eval_list.append(item)

    loader_tuple = ((
        lambda: None,
        lambda data_list=eval_list: build_dataloader(cfg, data_list, shuffle=False, train=False, mreserve=mreserve)
    ),)

    return loader_tuple, lid2str, lid2int, None

def build_dataloader_list(cfg, mreserve=False):
    if cfg.running.data_name == "esc50":
        return build_dataloader_list_esc50(cfg, mreserve=mreserve)
    elif cfg.running.data_name == "UrbanSound8K":
        return build_dataloader_list_us8k(cfg, mreserve=mreserve)
    elif cfg.running.data_name == "audioset":
        return build_dataloader_list_audioset(cfg, mreserve=mreserve)
    else:
        raise ValueError(f"unrecognized dataset `{cfg.running.data_name}`.") 

