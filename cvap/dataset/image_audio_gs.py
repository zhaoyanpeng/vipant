import os
import io
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
from pydub import AudioSegment

import tensorflow as tf
import multiprocessing as mp
import torch.utils.data as data
import torch.nn.functional as F

from . import PairImageSpectrogramTFRecords
from .audio import (
    make_transform, _extract_kaldi_spectrogram
)

class ImageAudioDatasetNpzGS(data.Dataset):
    """ `__getitem__' loads .npz from disk (Google Storage).
    """
    def __init__(self, cfg, data_name, train):
        data_path = f"{cfg.data_root}/{data_name}.csv"
        assert tf.io.gfile.exists(data_path), f"{data_path} is not a file."
        self.dataset = list()
        with tf.io.gfile.GFile(data_path, "r") as fr:
            for iline, line in enumerate(fr):
                record = json.loads(line)
                self.dataset.append(record) 
                if not train and iline + 1 == cfg.eval_samples:
                    break
        if train and cfg.train_samples > 0. and cfg.train_samples < 1.:
            k = int(len(self.dataset) * cfg.train_samples)
            #self.dataset = np.random.choice(self.dataset, k, replace=False)
            shuffled_indice = np.random.permutation(np.random.permutation(len(self.dataset)))
            self.dataset = [self.dataset[i] for i in shuffled_indice[:k]]
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

        aclip_file = io.BytesIO(tf.io.gfile.GFile(aclip_file, "rb").read()) 
        frame_file = io.BytesIO(tf.io.gfile.GFile(frame_file, "rb").read()) 

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

class ImageAudioDatasetSrcGS(data.Dataset):
    """ `__getitem__' loads .npz from disk (Google Storage).
    """
    def __init__(self, cfg, data_name, train):
        data_path = f"{cfg.data_root}/{data_name}.csv"
        assert tf.io.gfile.exists(data_path), f"{data_path} is not a file."
        # sox_io is the default backend on linux, but it doesn't support Byte streams
        # soundfile supports Byte streams but doesn't support mp3, so we have to convert
        # .mp3 to .wav streams and set the default backend to soundfile
        torchaudio.set_audio_backend("soundfile") 
        self.dataset = list()
        with tf.io.gfile.GFile(data_path, "r") as fr:
            for iline, line in enumerate(fr):
                record = json.loads(line)
                self.dataset.append(record) 
                if not train and iline + 1 == cfg.eval_samples:
                    break
        if train and cfg.train_samples > 0. and cfg.train_samples < 1.:
            k = int(len(self.dataset) * cfg.train_samples)
            #self.dataset = np.random.choice(self.dataset, k, replace=False)
            shuffled_indice = np.random.permutation(np.random.permutation(len(self.dataset)))
            self.dataset = [self.dataset[i] for i in shuffled_indice[:k]]
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

        aclip_file = io.BytesIO(tf.io.gfile.GFile(aclip_file, "rb").read()) 
        frame_file = io.BytesIO(tf.io.gfile.GFile(frame_file, "rb").read()) 

        images = np.load(frame_file)
        images = [images[key] for key in images.files if len(images[key]) != 0]
        assert len(images) != 0, f"no frame exist: |images| = {len(images)}"
        if self.train:
            idx = np.random.choice(len(images), 1)[0]
        else:
            idx = int(np.ceil(len(images) / 2)) - 1
        image = images[idx]

        # mp3 -> wav byte stream
        wav_bytes = io.BytesIO()
        AudioSegment.from_file(aclip_file).export(wav_bytes, format="wav")
        aclip_file = wav_bytes
        
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

