import os, csv, json
import numpy as np
from omegaconf import DictConfig, OmegaConf
import hydra

import torchaudio
from torchvision.transforms import InterpolationMode, Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import Image as PILImage

from cvap.dataset import PairImageSpectrogramTFRecords
from cvap.util import seed_all_rng


# helpers

def index_to_dict(data_path, index_file):
    video_file = f"{data_path}/clip/{{}}.p0.mp4"
    audio_file = f"{data_path}/clip/{{}}.p0.mp3"
    frame_file = f"{data_path}/frame/{{}}.{{}}_{{:02}}.jpg" 
    dataset = {"video": [], "audio": [], "mframe": [], "rframe": []} 
    with open(index_file, 'r') as f:
        for line in f:
            ytid, meta = json.loads(line)
            
            nframe = meta[2]
            mframe = int(np.ceil(nframe / 2)) # median frame
            rframe = np.random.choice(range(1, nframe + 1), replace=False) # random frame 
            mframe_f = frame_file.format(ytid, meta[2], mframe) 
            rframe_f = frame_file.format(ytid, meta[2], rframe) 
            
            audio_f = audio_file.format(ytid)
            video_f = video_file.format(ytid)
            
            dataset["video"].append(video_f)
            dataset["audio"].append(audio_f)
            dataset["mframe"].append(mframe_f)
            dataset["rframe"].append(rframe_f)
    return dataset

# script

@hydra.main(config_path="configs")
def preprocess(cfg: DictConfig) -> None:
    dcfg = cfg.preprocessing.dataset
    seed_all_rng(dcfg.seed)
    voice_clips = index_to_dict(dcfg.data_root, f"{dcfg.csv_root}/{dcfg.idx_name}.idx") 

    def _extract_spectrogram(filename):
        waveform, sample_rate = torchaudio.load(f"{filename}")
        fbank_feat = torchaudio.compliance.kaldi.fbank(
            waveform, 
            sample_frequency=sample_rate, 
            num_mel_bins=dcfg.mel_bins,
            high_freq=8000,
            low_freq=0,
            use_log_fbank=True, 
            window_type="hamming",
        )
        fbank_feat = fbank_feat[:dcfg.max_audio_len]
        return fbank_feat.numpy()
    spectrograms = (_extract_spectrogram(clip) for clip in voice_clips["audio"])
    
    def _transform(n_px):
        return Compose([
            Resize(n_px, interpolation=InterpolationMode.BICUBIC),
            CenterCrop(n_px),
            lambda image: image.convert("RGB"),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
    T = _transform(dcfg.input_resolution)

    images = (T(PILImage.open(clip)).numpy() for clip in voice_clips[dcfg.frame_type])
    image_names = (os.path.basename(clip) for clip in voice_clips[dcfg.frame_type])
    save_path = dcfg.data_root + f"/tfrecord/{dcfg.idx_name}"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    record_name = f"{dcfg.input_resolution}.{dcfg.mel_bins}.tfrecord"
    save_path = f"{save_path}/{record_name}"
    PairImageSpectrogramTFRecords.write(spectrograms, images, image_names, fname=save_path)


if __name__ == "__main__":
    preprocess()

