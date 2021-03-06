import abc
import torch
import torchaudio
import numpy as np
from omegaconf.listconfig import ListConfig
from omegaconf.dictconfig import DictConfig

import torchvision.transforms as transforms
from torchvision.transforms import Compose, ToTensor, Normalize
from torchaudio.transforms import FrequencyMasking, TimeMasking

def _extract_kaldi_spectrogram(
    filename, params, train=True, mean_channel=False, zero_mean_wf=False, max_audio_len=1000, transform_audio=None, tile_audio=False,
):
    waveform, sample_rate = torchaudio.load(filename)
    if mean_channel: # mean along channel # TODO else branch should take a specific channel
        waveform = waveform.mean(0, keepdim=True)
    desired_len = int((max_audio_len / 100) * sample_rate)
    if tile_audio and desired_len > waveform.shape[-1]:
        ntile = int(np.ceil(desired_len / waveform.shape[-1]))
        waveform = torch.tile(waveform, (1, ntile))[:desired_len]
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

def make_transform(cfg):
    transform_fbank = transform_audio = None
    if cfg.transform_audio:
        tfm_list = list()
        for name, params in cfg.audio_transforms:
            if isinstance(params, DictConfig): 
                tfm_list.append(eval(name)(**params))
            else:
                tfm_list.append(eval(name)(*params))
        if len(tfm_list) > 0: 
            transform_audio = Compose(tfm_list) 
    if cfg.transform_fbank:
        tfm_list = list()
        for name, params in cfg.fbank_transforms:
            if isinstance(params, DictConfig): 
                tfm_list.append(eval(name)(**params))
            else:
                tfm_list.append(eval(name)(*params))
        if len(tfm_list) > 0: 
            tfm_list = [lambda x: x.T, ToTensorKeepdim()] + tfm_list + [lambda x: x.T]
            transform_fbank = Compose(tfm_list) 
    #print(transform_audio, transform_fbank)
    return transform_audio, transform_fbank

class ToTensorKeepdim(ToTensor):
    def __call__(self, x):
        if isinstance(x, torch.Tensor):
            return x
        x = super(ToTensorKeepdim, self).__call__(x[..., None])
        return x.squeeze_(0)

class AbstractTransform(abc.ABC):
    @abc.abstractmethod
    def __call__(self, x):
        pass
    def __repr__(self):
        return self.__class__.__name__ + '()'

class RandomFlip(AbstractTransform):
    def __init__(self, p=0.5):
        super(RandomFlip, self).__init__()
        self.p = p

    @staticmethod
    def random_flip(x, p):
        if x.dim() > 2:
            flip_mask = torch.rand(x.shape[0], device=x.device) <= p
            x[flip_mask] = x[flip_mask].flip(-1)
        else:
            if torch.rand(1) <= p:
                x = x.flip(-1)
        return x

    def __call__(self, x):
        return self.random_flip(x, self.p)
       
class RandomScale(AbstractTransform):
    def __init__(self, scale=1.5, keep_len=False):
        super(RandomScale, self).__init__()
        self.scale = scale
        self.keep_len = keep_len

    @staticmethod
    def random_scale(x, scale, keep_len):
        scaling = np.power(scale, np.random.uniform(-1, 1))
        output_len = int(x.shape[-1] * scaling)
        base = torch.arange(output_len, device=x.device, dtype=x.dtype).div_(scaling)

        ref1 = base.clone().type(torch.int64)
        ref2 = torch.min(ref1 + 1, torch.full_like(ref1, x.shape[-1] - 1, dtype=torch.int64))
        r = base - ref1.type(base.type())
        scaled_x = (1 - r) * x[..., ref1] + r * x[..., ref2]
        if keep_len:
            scaled_x = RandomCrop.random_crop(scaled_x, x.shape[-1], True) # keep the same length
        return scaled_x

    def __call__(self, x):
        return self.random_scale(x, self.scale, self.keep_len)

class RandomCrop(AbstractTransform):
    def __init__(self, output_len=44100, train=True):
        super(RandomCrop, self).__init__()
        self.output_len = output_len
        self.train = train

    @staticmethod
    def random_crop(x, output_len, train):
        if x.shape[-1] <= output_len:
            return x
        if train:
            left = np.random.randint(0, x.shape[-1] - output_len)
        else: # center
            left = int(round(0.5 * (x.shape[-1] - output_len)))

        old_std = x.float().std() * 0.5
        cropped_x = x[..., left : left + output_len]

        new_std = cropped_x.float().std()
        if new_std < old_std:
            cropped_x = x[..., : output_len]

        out_std = cropped_x.float().std()
        if old_std > new_std > out_std:
            cropped_x = x[..., -output_len:]
        return cropped_x

    def __call__(self, x):
        return self.random_crop(x, self.output_len, self.train)  

class RandomPad(AbstractTransform):
    def __init__(self, output_len=88200, train=True, padding_value=None):
        super(RandomPad, self).__init__()
        self.output_len = output_len
        self.train = train
        self.padding_value = padding_value

    @staticmethod
    def random_pad(x, output_len, train, padding_value=None):
        if x.shape[-1] >= output_len: 
            return x
        if train:
            left = np.random.randint(0, output_len - x.shape[-1])
        else: # center
            left = int(round(0.5 * (output_len - x.shape[-1])))

        right = output_len - (left + x.shape[-1])
        if padding_value is not None:
            pad_value_left = pad_value_right = padding_value 
        else: # mean over channel? 
            pad_value_left = x[..., 0].float().mean().to(x.dtype)
            pad_value_right = x[..., -1].float().mean().to(x.dtype)
        padded_x = torch.cat((
            torch.zeros(x.shape[:-1] + (left,), dtype=x.dtype, device=x.device).fill_(pad_value_left),
            x,
            torch.zeros(x.shape[:-1] + (right,), dtype=x.dtype, device=x.device).fill_(pad_value_right)
        ), dim=-1)
        return padded_x

    def __call__(self, x):
        return self.random_pad(x, self.output_len, self.train, self.padding_value)  

class RandomNoise(AbstractTransform):
    def __init__(self, snr_min_db=10.0, snr_max_db=120.0, p=0.25):
        super(RandomNoise, self).__init__()
        self.snr_min_db = snr_min_db
        self.snr_max_db = snr_max_db
        self.p = p

    @staticmethod
    def random_noise(x, snr_min_db, snr_max_db, p):
        if np.random.rand() > p:
            return x
        target_snr = np.random.rand() * (snr_max_db - snr_min_db + 1.0) + snr_min_db

        x_watts = torch.mean(x ** 2, dim=(-1, -2))
        x_db = 10 * torch.log10(x_watts)

        noise_db = x_db - target_snr
        noise_watts = 10 ** (noise_db / 10) + 1e-7
        noise = torch.normal(0.0, noise_watts.item() ** 0.5, x.shape)

        noise_x = x + noise
        return noise_x

    def __call__(self, x):
        return self.random_noise(x, self.snr_min_db, self.snr_max_db, self.p) 

class SimpleRandomNoise(AbstractTransform):
    def __init__(self, scale=10.0, shift=10, p=0.25):
        super(SimpleRandomNoise, self).__init__()
        self.scale = scale 
        self.shift = shift
        self.p = p

    @staticmethod
    def random_noise(x, scale, shift, p):
        # expect a 2d tensor
        if np.random.rand() > p:
            return x
        noise_x = x + torch.rand(x.shape) * np.random.rand() / scale 
        noise_x = torch.roll(noise_x, np.random.randint(-shift, shift), -1)
        return noise_x

    def __call__(self, x):
        return self.random_noise(x, self.scale, self.shift, self.p) 

class FbankTransform:
    def __init__(self):
        self.transform = transforms.Compose([
            lambda x: x.T,
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[-4.93839311], std=[5.75751113]
            ),
            FrequencyMasking(48),
            TimeMasking(300),
            lambda x: x.transpose(-1, -2)
        ])
        self.transform_prime = transforms.Compose([
            lambda x: x.T,
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[-4.93839311], std=[5.75751113]
            ),
            FrequencyMasking(32),
            TimeMasking(200),
            lambda x: x.transpose(-1, -2)
        ])
        self.transform_eval = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[-4.93839311], std=[5.75751113]
            ),
        ])

    def __call__(self, x, both, train):
        if not train:
            return self.transform_eval(x), np.array([[[1]]])
        else:
            y1 = self.transform_prime(x)
            y2 = self.transform(x) if both else np.array([[[1]]])
            return y1, y2
