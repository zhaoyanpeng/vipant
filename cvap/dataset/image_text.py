import os
import csv
import glob
import json
import torch
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

import multiprocessing as mp
import torch.utils.data as data
import torch.nn.functional as F

from .ast import make_image_transform
from .audio import (
    make_transform, _extract_kaldi_spectrogram 
)
from .audio_text import (
    build_dataloader, build_audiocaps_data_list, AudioTextDatasetSrc
)
from clip import tokenize

class ImageTextDatasetSrc(AudioTextDatasetSrc):
    """ `__getitem__' loads raw file from disk.
    """
    def __init__(self, cfg, data_list, train):
        super().__init__(cfg, data_list, train)
        self.frame_key = cfg.frame_key
        self.transform_image = make_image_transform(cfg.resolution)

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

    def __getitem__(self, index):
        akey = self.aclip_key
        fkey = self.frame_key
        name = self.dataset[index]["id"]
        sub_dir = self.dataset[index]["dir"]
        label_str = self.dataset[index]["label_str"]
        label_int = self.dataset[index]["label_int_bpe"]
        aclip = self.dataset[index][akey][0]
        frame = images = self.dataset[index][fkey]

        sub_dir = "" if len(sub_dir) == 0 else f"{sub_dir}/"
        aclip = aclip if aclip == name else f"{akey}/{name}.{aclip}"
        aclip_file = f"{self.cfg.data_root}/{sub_dir}{aclip}"
        
        # image
        frame_emb_file = None
        if isinstance(frame, str):
            frame_file = f"{self.cfg.data_root}/{sub_dir}{fkey}/{name}.{frame}"
        else:
            idx = np.random.choice(len(images), 1)[0] if self.train else int(np.ceil(len(images) / 2)) - 1
            frame_file = f"{self.cfg.data_root}/{sub_dir}{fkey}/{name}.{images[idx]}"
            if self.cfg.frame_emb is not None:
                frame_emb_file = f"{self.cfg.data_root}/{self.cfg.frame_emb}/{name}.{images[idx].rsplit('.', 1)[0]}.npz"
        # higher priority for pre-computed frame embeddings
        image = self._image2embed(frame_emb_file) if frame_emb_file is not None else self._image2numpy(frame_file)
        
        # audio
        audio = self._audio2numpy_cst(aclip_file)

        if not self.cfg.audio.eval_norms and len(self.audio_norms) == 2:
            mean, std = self.audio_norms
            audio = (audio - mean) / std

        #if self.train and self.transform_fbank is not None:
        if not self.cfg.audio.eval_norms and self.train and self.transform_fbank is not None:
            audio = self.transform_fbank(audio)

        if self.train:
            idx = np.random.choice(len(label_int), 1)[0]
            text = label_int[idx]
        else:
            text = label_int

        audio = audio[None]
        image = image[None]
        item = {"image": image, "audio": audio, "text": text, "name": name}
        return item 

    def __len__(self):
        return self.length

class ImageTextCollator:
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
            #name = list(itertools.chain.from_iterable(name))
        else:
            raise ValueError(f"unrecognized `{type(text_list[0][0])}`")
        # https://stackoverflow.com/a/38619333
        text = np.array(list(itertools.zip_longest(*text_list, fillvalue=0))).T
        return (
            np.concatenate(union["image"], axis=0),
            text,
            name,
        )

def build_dataloader_audiocaps(cfg, data_name, shuffle=True, train=True):
    name_list = data_name.split(",")
    dataset = list()
    for name in name_list:
        subset = build_audiocaps_data_list(cfg.running, name)
        dataset.extend(subset)
    return build_dataloader(cfg, dataset, ImageTextDatasetSrc, shuffle=shuffle, train=train, collator_cls=ImageTextCollator)

def build_image_text_dataloader(cfg, data_name, *args, shuffle=True, train=True, **kwargs):
    if data_name.startswith("audiocaps"): # can only do w/ AudioCaps
        return build_dataloader_audiocaps(
            cfg, data_name, shuffle=shuffle, train=train
        )
    else:
        raise ValueError(f"unrecognized dataset `{data_name}`.") 

