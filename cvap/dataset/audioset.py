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
from clip import tokenize

class AudiosetDatasetNpz(data.Dataset):
    """ `__getitem__' loads .npz from disk.
    """
    def __init__(self, cfg, data_name, train, label_map):
        data_path = f"{cfg.data_root}/{data_name}.csv"
        assert os.path.isfile(data_path), f"{data_path} is not a file."
        self.label_map = label_map
        self.num_label = len(label_map)
        self.dataset = list()
        with open(data_path, "r") as fr:
            for iline, line in enumerate(fr):
                record = json.loads(line)
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
        if self.train and self.transform_fbank is not None:
            audio = self.transform_fbank(audio)
        npad = max_audio_len - audio.shape[0]
        if npad > 0:
            audio = np.pad(audio, ((0, npad), (0, 0)), "constant", constant_values=(0., 0.))
        
        image = image[None]
        audio = audio[None]
        
        # TODO return all labels if it is a classification task
        category = categories[ict]
        label, _, text_int = self.label_map[category] 

        item = {"image": image, "audio": audio, "text": text_int, "label": label, "name": name}
        return item 

    def __len__(self):
        return self.length

class AudiosetDatasetSrc(data.Dataset):
    """ `__getitem__' loads raw file from disk.
    """
    def __init__(self, cfg, data_name, train, label_map):
        data_path = f"{cfg.data_root}/{data_name}.csv"
        assert os.path.isfile(data_path), f"{data_path} is not a file."
        self.label_map = label_map
        self.num_label = len(label_map)
        self.dataset = list()
        with open(data_path, "r") as fr:
            for iline, line in enumerate(fr):
                record = json.loads(line)
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
        akey = "aclip"
        fkey = "frame_224"
        dir = self.dataset[index]["dir"] 
        name = self.dataset[index]["id"] 
        aclip = self.dataset[index][akey][0] 
        frame = self.dataset[index][fkey]
        categories = self.dataset[index]["labels"]

        aclip_file = f"{self.cfg.data_root}/{dir}/{akey}/{name}.{aclip}"
        frame_file = f"{self.cfg.data_root}/{dir}/{fkey}/{name}.{frame}"

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
        audio = _extract_kaldi_spectrogram(
            aclip_file, 
            self.kaldi_params, 
            train=self.train,
            max_audio_len=max_audio_len,
            transform_audio=(self.transform_audio if self.train else None)
        ) # (..., time, freq)
        
        if self.train and self.transform_fbank is not None:
            audio = self.transform_fbank(audio)

        npad = max_audio_len - audio.shape[0]
        if npad > 0:
            audio = np.pad(audio, ((0, npad), (0, 0)), "constant", constant_values=(0., 0.))
        
        image = image[None]
        audio = audio[None]

        # TODO return all labels if it is a classification task
        category = categories[ict]
        label, _, text_int = self.label_map[category] 

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
        if isinstance(text_list[0][0], int): # train
            label = np.array(label)
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
        dataset = AudiosetDatasetSrc(rcfg, data_name, train, label_map)
    elif data_name.startswith("npz"):
        dataset = AudiosetDatasetNpz(rcfg, data_name, train, label_map)
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
