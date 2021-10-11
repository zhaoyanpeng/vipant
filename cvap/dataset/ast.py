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

def make_image_transform(n_px):
    return Compose([
        Resize(n_px, interpolation=InterpolationMode.BICUBIC),
        CenterCrop(n_px),
        lambda image: image.convert("RGB"),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

class ASTNpz(data.Dataset):
    """ `__getitem__' loads raw file from disk. The same inputs as ASTSrc.
    """
    def __init__(self, cfg, data_name, train, label_map, weighted):
        data_path = f"{cfg.data_root}/{data_name}.csv"
        assert os.path.isfile(data_path), f"{data_path} is not a file."
        self.cfg = cfg
        self.label_map = label_map
        self.num_label = len(label_map)
        label_counts = np.zeros(self.num_label)
        self.dataset = list()
        with open(data_path, "r") as fr:
            for iline, line in enumerate(fr):
                record = json.loads(line)
                if cfg.cat_label:
                    self._cat_label(record)
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
        # compatible with AudioSet and Balanced AudioSet
        self.aclip_key = "aclip_128"
        self.frame_key = "frame_224"
        self.train = train

        acfg = cfg.audio
        self.transform_image = make_image_transform(cfg.resolution)
        self.transform_audio, self.transform_fbank = make_transform(acfg)

    def _shuffle(self):
        pass

    def _cat_label(self, record):
        categories = record["labels"]
        label_text = [re.sub(
            f"^{self.cfg.prompt}", "", self.label_map[category][1]
        ).strip() for category in categories]
        label_text = self.cfg.prompt + " " +  ", ".join(label_text)
        record["captions"] = [label_text]
        record["captions_bpe"] = tokenize(
            record["captions"], as_list=True
        ) # add bpe captions

    def _image2numpy(self, fname):
        if fname is not None:
            try:
                images = np.load(fname)
                images = [images[key] for key in images.files if len(images[key]) != 0]
                idx = np.random.choice(len(images), 1)[0] if self.train else int(np.ceil(len(images) / 2)) - 1
                image = images[idx]
            except Exception as e:
                h = w = self.cfg.resolution
                image = PILImage.fromarray(
                    (np.random.rand(h, w, 3) * 256).astype(np.uint8)
                )
                warnings.warn(f"use random image instead because `{e}` {fname}.")
                image = self.transform_image(image).cpu().numpy()
        else:
            image = np.array([[[1]]])
        return image

    def _process_item(self, index):
        akey = self.aclip_key
        fkey = self.frame_key
        sub_dir = self.dataset[index]["dir"]
        name = self.dataset[index]["id"]
        aclip = self.dataset[index][akey]
        frame = images = self.dataset[index][fkey]
        categories = self.dataset[index]["labels"]

        sub_dir = "" if len(sub_dir) == 0 else f"{sub_dir}/"
        aclip_file = f"{self.cfg.data_root}/{sub_dir}{akey}/{name}.{aclip}"
        frame_file = f"{self.cfg.data_root}/{sub_dir}{fkey}/{name}.{frame}" if self.cfg.imagine else None

        if not self.cfg.clf: # dummy
            if self.cfg.cat_label:
                record = self.dataset[index]
                label, text, text_int = 0, record["captions"][0], record["captions_bpe"][0]
            else:
                ict = np.random.choice(len(categories), 1)[0] if self.train else 0
                category = categories[ict]
                label, text, text_int = self.label_map[category]
        else: # classification task
            label_set = set([self.label_map[category][0] for category in categories])
            label = [1 if i in label_set else 0 for i in range(self.num_label)]
            text_int = [0] # TODO concatenate all text pieces

        item = {"text": text_int, "name": name}
        return item, label, aclip_file, frame_file

    def __getitem__(self, index):
        item, label, aclip_file, frame_file = self._process_item(index)

        max_audio_len = self.cfg.max_audio_len
        audio = np.load(aclip_file)["flag"] # (..., time, freq): `flag' is used as the key accidentally
        image = self._image2numpy(frame_file)

        npad = max_audio_len - audio.shape[0]
        if npad > 0:
            audio = np.pad(audio, ((0, npad), (0, 0)), "constant", constant_values=(0., 0.))

        if not self.cfg.audio.eval_norms and len(self.audio_norms) == 2:
            mean, std = self.audio_norms
            audio = (audio - mean) / std

        #if self.train and self.transform_fbank is not None:
        if not self.cfg.audio.eval_norms and self.train and self.transform_fbank is not None:
            audio = self.transform_fbank(audio)

        image = image[None]
        audio = audio[None]
        item.update({"image": image, "audio": audio, "label": label})
        return item

    def __len__(self):
        return self.length

class ASTSrc(data.Dataset):
    """ `__getitem__' loads raw file from disk.
    """
    def __init__(self, cfg, data_name, train, label_map, weighted, filter_set=None):
        data_path = f"{cfg.data_root}/{data_name}.csv"
        assert os.path.isfile(data_path), f"{data_path} is not a file."
        self.cfg = cfg
        self.label_map = label_map
        self.num_label = len(label_map)
        label_counts = np.zeros(self.num_label) 
        self.dataset = list()
        with open(data_path, "r") as fr:
            for iline, line in enumerate(fr):
                record = json.loads(line)
                if filter_set is not None and record["id"] not in filter_set:
                    continue # let us skip this sample
                if cfg.cat_label:
                    self._cat_label(record)
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
        # compatible with AudioSet and Balanced AudioSet
        self.aclip_key = "clip" if "clip" in self.dataset[0] else "aclip"
        self.frame_key = cfg.frame_key
        self.train = train
        
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

    def _cat_label(self, record):
        categories = record["labels"]
        label_text = [re.sub(
            f"^{self.cfg.prompt}", "", self.label_map[category][1]
        ).strip() for category in categories]
        label_text = self.cfg.prompt + " " +  ", ".join(label_text)
        record["captions"] = [label_text]
        record["captions_bpe"] = tokenize(
            record["captions"], as_list=True
        ) # add bpe captions

    def _process_item(self, index):
        akey = self.aclip_key
        fkey = self.frame_key
        sub_dir = self.dataset[index]["dir"] 
        name = self.dataset[index]["id"] 
        aclip = self.dataset[index][akey][0] 
        frame = images = self.dataset[index][fkey]
        categories = self.dataset[index]["labels"]
        
        sub_dir = "" if len(sub_dir) == 0 else f"{sub_dir}/"
        aclip_file = f"{self.cfg.data_root}/{sub_dir}{akey}/{name}.{aclip}"

        frame_file = frame_emb_file = None
        if self.cfg.imagine:
            if isinstance(frame, str):
                frame_file = f"{self.cfg.data_root}/{sub_dir}{fkey}/{name}.{frame}"
            else:
                idx = np.random.choice(len(images), 1)[0] if self.train else int(np.ceil(len(images) / 2)) - 1
                frame_file = f"{self.cfg.data_root}/{sub_dir}{fkey}/{name}.{images[idx]}"
                if self.cfg.frame_emb is not None:
                    frame_emb_file = f"{self.cfg.data_root}/{self.cfg.frame_emb}/{name}.{images[idx].rsplit('.', 1)[0]}.npz"

        if not self.cfg.clf:
            if self.cfg.cat_label:
                record = self.dataset[index]
                label, text, text_int = 0, record["captions"][0], record["captions_bpe"][0]
            else:
                ict = np.random.choice(len(categories), 1)[0] if self.train else 0
                category = categories[ict]
                label, text, text_int = self.label_map[category]
        else: # classification task
            label_set = set([self.label_map[category][0] for category in categories])
            label = [1 if i in label_set else 0 for i in range(self.num_label)] 
            text_int = [0] # TODO concatenate all text pieces

        item = {"text": text_int, "name": name}
        return item, label, aclip_file, frame_file, frame_emb_file

    def _image2embed(self, fname):
        try:
            image = np.load(fname)["v"]
        except Exception as e:
            image = np.random.rand(self.cfg.embed_dim).astype("float32")
            warnings.warn(f"use random image instead because `{e}` {fname}.")
        return image

    def _image2numpy(self, fname):
        if fname is not None:
            try:
                if fname.endswith(".npz"):
                    images = np.load(fname)
                    images = [images[key] for key in images.files if len(images[key]) != 0]
                    idx = np.random.choice(len(images), 1)[0] if self.train else int(np.ceil(len(images) / 2)) - 1
                    image = images[idx]
                else:
                    image = PILImage.open(fname)
                    image = self.transform_image(image).cpu().numpy()
            except Exception as e:
                h = w = self.cfg.resolution
                image = PILImage.fromarray(
                    (np.random.rand(h, w, 3) * 256).astype(np.uint8)
                )
                warnings.warn(f"use random image instead because `{e}` {fname}.")
                image = self.transform_image(image).cpu().numpy()
        else:
            image = np.array([[[1]]])
        return image

    def _audio2numpy_clf(self, aclip_file, label):
        wf, sr = torchaudio.load(aclip_file)
        wf = wf[:1] #wf.mean(0, keepdim=True)
        wf = wf - wf.mean()

        sampler = np.random if self.cfg.np_rnd else random 

        #if self.train and sampler.random() < self.cfg.mixup_rate:
        if not self.cfg.audio.eval_norms and self.train and sampler.random() < self.cfg.mixup_rate:
            idx_mix = sampler.randint(0, self.length if self.cfg.np_rnd else self.length - 1)
            _, label_mix, aclip_file, _, _ = self._process_item(idx_mix)
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
        return audio

    def _audio2numpy_cst(self, aclip_file):
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

        npad = max_audio_len - audio.shape[0]
        if npad > 0:
            audio = np.pad(audio, ((0, npad), (0, 0)), "constant", constant_values=(0., 0.))
        return audio

    def __getitem__(self, index):
        item, label, aclip_file, frame_file, frame_emb_file = self._process_item(index)

        # higher priority for pre-computed frame embeddings
        image = (self._image2embed(frame_emb_file)
            if frame_emb_file is not None and self.cfg.imagine else self._image2numpy(frame_file)
        )
        audio = (self._audio2numpy_clf(aclip_file, label)
            if self.cfg.clf else self._audio2numpy_cst(aclip_file)
        )

        if not self.cfg.audio.eval_norms and len(self.audio_norms) == 2:
            mean, std = self.audio_norms
            audio = (audio - mean) / std

        #if self.train and self.transform_fbank is not None:
        if not self.cfg.audio.eval_norms and self.train and self.transform_fbank is not None:
            audio = self.transform_fbank(audio)

        image = image[None]
        audio = audio[None]
        item.update({"image": image, "audio": audio, "label": label})
        return item 

    def __len__(self):
        return self.length
