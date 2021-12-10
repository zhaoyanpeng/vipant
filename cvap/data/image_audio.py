import os
import glob
import json
import torch
import warnings
import torchaudio
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image as PILImage
from itertools import cycle, islice, chain
from einops import rearrange, repeat

import multiprocessing as mp
import torch.utils.data as data
import torch.nn.functional as F

from .audio import (
    make_transform, _extract_kaldi_spectrogram, FbankTransform
)
from .image import make_clip_image_transform as make_image_transform
from .image import BarlowImageTransform as ImageTransform
#from .image_audio_gs import (
#    ImageAudioDatasetNpzGS, ImageAudioDatasetSrcGS
#)

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
            #self.dataset = np.random.choice(self.dataset, k, replace=False)
            shuffled_indice = np.random.permutation(np.random.permutation(len(self.dataset)))
            self.dataset = [self.dataset[i] for i in shuffled_indice[:k]]
        self.audio_norms = cfg.audio.norms
        # compatible with AudioSet and Balanced AudioSet
        self.aclip_key = "clip" if "clip" in self.dataset[0] else "aclip"
        self.frame_key = cfg.frame_key
        self.length = len(self.dataset)
        self.train = train
        self.cfg = cfg
        
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

    def _process_item(self, index):
        akey = self.aclip_key
        fkey = self.frame_key
        sub_dir = self.dataset[index]["dir"]
        name = self.dataset[index]["id"] 
        aclip = self.dataset[index][akey][0] 
        frame = images = self.dataset[index][fkey]

        sub_dir = "" if len(sub_dir) == 0 else f"{sub_dir}/"
        aclip_file = f"{self.cfg.data_root}/{sub_dir}{akey}/{name}.{aclip}"

        frame_emb_file = None
        if isinstance(frame, str):
            frame_file = f"{self.cfg.data_root}/{sub_dir}{fkey}/{name}.{frame}"
        else:
            idx = np.random.choice(len(images), 1)[0] if self.train else int(np.ceil(len(images) / 2)) - 1
            frame_file = f"{self.cfg.data_root}/{sub_dir}{fkey}/{name}.{images[idx]}"
            if self.cfg.frame_emb is not None:
                frame_emb_file = f"{self.cfg.data_root}/{self.cfg.frame_emb}/{name}.{images[idx].rsplit('.', 1)[0]}.npz"

        return name, aclip_file, frame_file, frame_emb_file

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

    def _audio2numpy(self, aclip_file):
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

        if not self.cfg.audio.eval_norms and len(self.audio_norms) == 2:
            mean, std = self.audio_norms
            audio = (audio - mean) / std

        #if self.train and self.transform_fbank is not None:
        if not self.cfg.audio.eval_norms and self.train and self.transform_fbank is not None:
            audio = self.transform_fbank(audio)
        return audio

    def __getitem__(self, index):
        name, aclip_file, frame_file, frame_emb_file = self._process_item(index)

        # higher priority for pre-computed frame embeddings
        image = self._image2embed(frame_emb_file) if frame_emb_file is not None else self._image2numpy(frame_file)
        audio = self._audio2numpy(aclip_file)

        image = image[None]
        audio = audio[None]
        item = {"image": image, "audio": audio, "name": name}
        return item

    def __len__(self):
        return self.length

class ImageAudioDatasetSiameseSrc(ImageAudioDatasetSrc):
    """ create two views of an image (audio).
        self.cfg.frame_emb is expected to be not None.
    """
    def __init__(self, cfg, data_name, train):
        super().__init__(cfg.running, data_name, train)
        self.lcfg = cfg.model.loss
        assert self.cfg.frame_emb is not None, f"`frame_emb` is None"
        if not cfg.running.clip_tf:
            from .image import BarlowImageTransform as ImageTransform
        else: # use `CLIPImageTransform` to generate multi-view images
            from .image import CLIPImageTransform as ImageTransform
            from .image import AuthenticCLIPImageTransform as ImageTransform
        self.transform_image = ImageTransform(self.cfg.resolution)
        self.transform_audio = None
        self.transform_fbank = FbankTransform()

    def _image2numpy(self, fname):
        if fname is not None:
            try:
                image = PILImage.open(fname)
            except Exception as e:
                h = w = self.cfg.resolution
                image = PILImage.fromarray(
                    (np.random.rand(h, w, 3) * 256).astype(np.uint8)
                )
                warnings.warn(f"use random image instead because `{e}` {fname}.")
            images = self.transform_image(image, self.lcfg.vv, self.train)
        else:
            images = (np.array([[[1]]]),) * 2
        return images

    def _audio2numpy(self, aclip_file):
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

        """
        if not self.cfg.audio.eval_norms and len(self.audio_norms) == 2:
            mean, std = self.audio_norms
            audio = (audio - mean) / std

        #if self.train and self.transform_fbank is not None:
        if not self.cfg.audio.eval_norms and self.train and self.transform_fbank is not None:
            audio = self.transform_fbank(audio)
        return audio[None], np.array([[[1]]])
        """

        audios = self.transform_fbank(audio, self.lcfg.aa, self.train)
        return audios

    def __getitem__(self, index):
        name, aclip_file, frame_file, frame_emb_file = self._process_item(index)

        image = self._image2embed(frame_emb_file) if self.lcfg.vp or self.lcfg.ap else np.array([[[1]]])
        images = tuple(x[None] for x in self._image2numpy(frame_file))
        audios = tuple(x[None] for x in self._audio2numpy(aclip_file))

        #image = np.array([[[1]]])
        #images = (self._image2numpy(frame_file), np.array([[[1]]]))
        #images = tuple(x[None] for x in images)
        #audios = (self._audio2numpy(aclip_file)[None], np.array([[[1]]]))
        #audios = tuple(x[None] for x in audios)

        item = {
            "image": image[None], "name": name,
            "image_v1": images[0], "image_v2": images[1],
            "audio_v1": audios[0], "audio_v2": audios[1]
        }
        return item

class ImageAudioCollator:
    def __init__(self, device=torch.device("cpu")):
        # RuntimeError: cannot pin 'torch.cuda.FloatTensor' only dense CPU tensors can be pinned
        # when pin_memory is true, the collator has to return CPU tensors
        self.device = device

    def __call__(self, records):
        union = { 
            k: [record.get(k) for record in records] for k in set().union(*records) 
        } 
        if "image_v1" in union:
            return (
                np.concatenate(union["image"], axis=0),
                np.concatenate(union["image_v1"], axis=0),
                np.concatenate(union["image_v2"], axis=0),
                np.concatenate(union["audio_v1"], axis=0),
                np.concatenate(union["audio_v2"], axis=0),
                union["name"],
            )
        else:
            return (
                np.concatenate(union["image"], axis=0),
                np.concatenate(union["audio"], axis=0),
                union["name"],
            )

def build_image_audio_dataloader(cfg, data_name, shuffle=True, train=True):
    ddp_mode = torch.distributed.is_initialized()
    rcfg = cfg.running
    from_gs = rcfg.data_root.startswith("gs://")
    if data_name.startswith("src"):
        if not from_gs:
            if not getattr(rcfg, "multi_view", False):
                dataset = ImageAudioDatasetSrc(rcfg, data_name, train)
            else:
                dataset = ImageAudioDatasetSiameseSrc(cfg, data_name, train)
        else:
            #dataset = ImageAudioDatasetSrcGS(rcfg, data_name, train)
            raise ValueError(f"unrecognized dataset `{data_name}`.")
    elif data_name.startswith("npz"):
        if not from_gs:
            dataset = ImageAudioDatasetNpz(rcfg, data_name, train)
        else:
            #dataset = ImageAudioDatasetNpzGS(rcfg, data_name, train)
            raise ValueError(f"unrecognized dataset `{data_name}`.")
    else:
        raise ValueError(f"unrecognized dataset `{data_name}`.")
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
