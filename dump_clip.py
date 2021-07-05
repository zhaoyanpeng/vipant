import multiprocessing
import subprocess
import threading
import requests
import datetime
import time, re
import math, os
import argparse

import torchaudio
import numpy as np
from collections import defaultdict
from torchvision.transforms import InterpolationMode, Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import Image as PILImage
from google.cloud import storage

storage_client = storage.Client()
bucket = storage_client.bucket("yannaudioset")

from clip import load 
from cvap.utils import seed_all_rng

seed_all_rng(1213)

# Data path options
parser = argparse.ArgumentParser()
parser.add_argument('--home', default='/home/yanpengz', type=str, help='')
parser.add_argument('--csv_root', default='./csv', type=str, help='')
parser.add_argument('--keepdata', default=False, type=bool, help='')
parser.add_argument('--check_gs', default=True, type=bool, help='')
parser.add_argument('--nprocess', default=1, type=int, help='')
parser.add_argument('--peeprate', default=100000, type=int, help='')
parser.add_argument('--portion', default="unbalanced", type=str, help='')
parser.add_argument('--chunk_b', default=0, type=int, help='')
parser.add_argument('--chunk_e', default=3000000, type=int, help='')
parser.add_argument('--resolution', default=224, type=int, help='image w & h')
parser.add_argument('--n_mel_bins', default=128, type=int, help='mel feature dim')
parser.add_argument('--src_url_base', default="gs://rowanaudioset/youtube_dump", type=str, help='')
parser.add_argument('--tgt_url_base', default="gs://yannaudioset", type=str, help='')
parser.add_argument('--clip_model_root', default="/net/nfs2.mosaic/yann/model/clip", type=str, help='')
parser.add_argument('--clip_model_name', default="ViT-B32", type=str, help='')
cfg = parser.parse_args()
# constants
home = cfg.home 
keepdata = cfg.keepdata
csv_root = cfg.csv_root 
suffix = f"{cfg.chunk_b}_{cfg.chunk_e}" 
# save paths 
part = f"{cfg.portion}_{suffix}" 
vroot = f"{home}/data/{part}/video"
froot = f"{home}/data/{part}/frame"
croot = f"{home}/data/{part}/vclip"
aroot = f"{home}/data/{part}/aclip"
audio_np_root = f"{home}/data/{part}/aclip_{cfg.n_mel_bins}"
frame_np_root = f"{home}/data/{part}/frame_{cfg.resolution}"
frame_en_root = f"{home}/data/{part}/frame_{cfg.clip_model_name}"
cfg.tgt_url_base = f"{cfg.tgt_url_base}/{part}" # 
# youtube ids
csv_all = ["balanced_train_segments.csv", "eval_segments.csv", "unbalanced_train_segments.csv"]
csv_balanced = ["balanced_train_segments.csv", "eval_segments.csv"]
csv_unbalanced = ["unbalanced_train_segments.csv"]
# save files
err_file = f"{home}/data/{part}/err_ytid.csv"

def _load_clip_model(cfg):
    model, _ = load(
        cfg.clip_model_name, cfg.clip_model_root, device="cpu", jit=False
    )
    model.train(False)
    return model
#clip_model = _load_clip_model(cfg)

def _build_name_set(prefix):
    def build_set(bbs):
        ytids = set()
        for x in bbs:
            ytid = x.name.rsplit("/", 1)[-1].split(".", 1)[0]
            ytids.add(ytid)
        return ytids
    aclip_list = storage_client.list_blobs('yannaudioset', prefix=f'{prefix}/aclip_128')
    frame_list = storage_client.list_blobs('yannaudioset', prefix=f'{prefix}/frame_224')
    
    set0 = build_set(aclip_list)
    set1 = build_set(frame_list)
    # print(len(set0), len(set1))
    return set0 & set1
if cfg.check_gs:
    #part = "all_0_35000"
    done_set = _build_name_set(part)
else:
    done_set = set()

def _transform(n_px):
    return Compose([
        Resize(n_px, interpolation=InterpolationMode.BICUBIC),
        CenterCrop(n_px),
        lambda image: image.convert("RGB"),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
T = _transform(cfg.resolution)

def _extract_spectrogram(filename, max_audio_len=1000, num_mel_bins=128):
    waveform, sample_rate = torchaudio.load(f"{filename}")
    fbank_feat = torchaudio.compliance.kaldi.fbank(
        waveform, 
        sample_frequency=sample_rate, 
        num_mel_bins=num_mel_bins,
        high_freq=8000,
        low_freq=0,
        use_log_fbank=True, 
        window_type="hamming",
    )
    fbank_feat = fbank_feat[:max_audio_len]
    return fbank_feat.numpy()

def prepare(cfg, verbose=False):
    paths = [
        vroot, froot, croot, aroot, audio_np_root, frame_np_root, frame_en_root
    ]
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)

    if verbose: 
        print(cfg)
        for name in paths + [err_file]:
            print(name)

    if cfg.portion == "unbalanced":
        return csv_unbalanced
    elif cfg.portion == "balanced":
        return csv_balanced
    else:
        return csv_all

def run(arg_list):
    ret = subprocess.run(
        " ".join(arg_list), capture_output=True, shell=True, text=True
    )
    return ret.stdout, ret.stderr, ret.returncode

def destroy(debug=False):
    if not debug:
        run(["rm", f"{vroot}/*"])
    run(["rm", f"{froot}/*"])
    run(["rm", f"{aroot}/*"])

def collect_ytid(csv_list):
    ids = defaultdict(list)
    nrow = 0
    for fname in csv_list:
        ifile = f"{csv_root}/{fname}"
        with open(ifile, "r") as fr:
            for _ in range(3):
                next(fr)
            for irow, row in enumerate(fr):
                row = row.split(", ")
                ids[row[0].strip()].append(
                    (row[1].strip(), row[2].strip(), row[3].strip().split(","))
                )
                nrow += 1
    print(f"total {nrow} examples.")
    return list(ids.keys()), ids

def skip_dl(ytid):
    name = ytid[0]  
    if not cfg.check_gs:
        return False
    url = cfg.tgt_url_base
    #url = cfg.tgt_url_base.rsplit("/", 1)[0] + "/all_0_35000"

    args = ["gsutil -q stat", f"{url}/{audio_np_root.split('/')[-1]}/{name}*"]
    _, _, c0 = run(args)
    #print(c0, " ".join(args))

    args = ["gsutil -q stat", f"{url}/{frame_np_root.split('/')[-1]}/{name}*"]
    _, _, c1 = run(args)
    #print(c1, " ".join(args))
    
    return c0 + c1 == 0

def dl_video(ytid):
    """ copy from cfg.src_url_base 
    :param ytid: (ytid, [(start_time, end_time, [label0, label1, ...])]) 
    """
    #if skip_dl(ytid):
    #    return "", "", 0, None
    name = ytid[0]  
    if name in done_set:
        #print(f"{name} done")
        return "", "", 0, None
    #print(f"{name} dl")
    #return "", "", 0, None
    gfile = f"{cfg.src_url_base}/{name}/{name}.mp4"
    ofile = f"{vroot}/{name}.mp4" 
    out, err, code = run(["gsutil -m", "cp", gfile, ofile])
    #print(f"\n-->>>\n|{code}|\n|{out.strip()}|\n|{err.strip()}|\n<<<--")
    ofile = ofile if code == 0 else None
    return out, err, code, ofile

def upload(ytid):
    """ send to cfg.tgt_url_base 
    :param ytid: (ytid, [(start_time, end_time, [label0, label1, ...])]) 
    """
    name = ytid[0]  

    local_audio_files = f"{aroot}/{name}*"
    run(["gsutil -m", "cp", local_audio_files, f"{cfg.tgt_url_base}/{aroot.split('/')[-1]}/"])
    #print(" ".join(["gsutil -m", "cp", local_audio_files, f"{cfg.tgt_url_base}/{aroot.split('/')[-1]}/"]))

    local_frame_files = f"{froot}/{name}*"
    run(["gsutil -m", "cp", local_frame_files, f"{cfg.tgt_url_base}/{froot.split('/')[-1]}/"])
    #print(" ".join(["gsutil -m", "cp", local_frame_files, f"{cfg.tgt_url_base}/{froot.split('/')[-1]}/"]))

    local_audio_npz = f"{audio_np_root}/{name}*" 
    run(["gsutil -m", "cp", local_audio_npz, f"{cfg.tgt_url_base}/{audio_np_root.split('/')[-1]}/"])
    #print(" ".join(["gsutil -m", "cp", local_audio_npz, f"{cfg.tgt_url_base}/{audio_np_root.split('/')[-1]}/"]))

    local_frame_npz = f"{frame_np_root}/{name}*" 
    run(["gsutil -m", "cp", local_frame_npz, f"{cfg.tgt_url_base}/{frame_np_root.split('/')[-1]}/"])
    #print(" ".join(["gsutil -m", "cp", local_frame_npz, f"{cfg.tgt_url_base}/{frame_np_root.split('/')[-1]}/"]))

def collect_clip(ytid, vfile):
    """
    :param ytid: (ytid, [(start_time, end_time, [label0, label1, ...])]) 
    """
    name = ytid[0] 
    b, e = [float(s) for s in ytid[1][0][:2]]
    # probe the length of the video
    out, err, code = run([
        "ffprobe", vfile, "-show_format 2>&1", "|", "sed -n -E 's/duration=([0-9]+).*/\\1/p'"
    ])
    m = float(out) # duration

    b, e = max(0, b), min(e, m)
    clips = [[b, e, "p0"]] # the gold clip

    # random background (non-event) clips
    # extract a clip from left & right of the event clip, respectively
    c, margin, min_len, nclip = 0, 3, 5, 2
    #commands = []
    b = b - margin
    e = e + margin
    for i in range(10000): # break when reaching the maximum # of clips
        # left side
        if b - 0 >= min_len:
            l = max(0, b - 10)
            t = b - l
            ss = str(datetime.timedelta(seconds=l))
            b = l - margin
            #print(ss, t)
            clips.append([l, l + t, f"n{c}"])
            c += 1
            if c >= nclip: break
        # right side
        if m - e >= min_len:
            l = e
            ss = str(datetime.timedelta(seconds=l))
            t = min(10, m - l)
            e = l + t + margin
            #print(ss, t)
            clips.append([l, l + t, f"n{c}"])
            c += 1
            if c >= nclip: break
        if (b - 0 < min_len and m - e < min_len) or (c > nclip):
            break # 
    return clips

def clip_audio(ytid, vfile, clips):
    """ extract audio clips
    :param ytid: (ytid, [(start_time, end_time, [label0, label1, ...])]) 
    """
    name = ytid[0] 
    main_out, main_err, main_code = None, None, 1
    for iclip, (b, e, flag) in enumerate(clips):
        fname = f"{name}.{flag}" 
        arg = [
            "ffmpeg -y", f"-i {vfile}", "-filter_complex", 
            f"\"[0:a]atrim=start={b}:end={e},asetpts=PTS-STARTPTS[b]\"",
            "-map '[b]'", "-strict -2", f"{aroot}/{fname}.mp3" 
        ] # https://superuser.com/a/723519
        #print(" ".join(arg))
        out, err, code = run(arg)
        if iclip == 0: # audio features
            main_out, main_err, main_code = out, err, code
            if os.path.isfile(f"{aroot}/{fname}.mp3"):
                np.savez_compressed(
                    f"{audio_np_root}/{fname}",
                    flag = _extract_spectrogram(
                        f"{aroot}/{fname}.mp3", num_mel_bins=cfg.n_mel_bins
                    )
                )
    return main_out, main_err, main_code

def collect_frame(ytid, vfile, clips, num_step=3): 
    """
    :param ytid: (ytid, [(start_time, end_time, [label0, label1, ...])]) 
    """
    name = ytid[0] 
    main_status = [False]
    for iclip, (b, e, flag) in enumerate(clips):
        len_step = (e  - b) / num_step 
        timestamps =  [b + i * len_step for i in range(num_step + 1)]
        commands = [[
            "ffmpeg -y", f"-ss {datetime.timedelta(seconds=l)}", f"-i {vfile}",  
            "-frames:v 1", "-q:v 1", f"{froot}/{name}.{flag}.{num_step + 1}_{i + 1:02}.jpg"
            ] for i, l in enumerate(timestamps) #enumerate([0, 6, 12]) #
        ]
        status = [False for _ in range(len(commands))]
        for i, arg in enumerate(commands):
            #print(" ".join(arg))
            out, err, code = run(arg)
            if code == 0:
                status[i] = True 
            #print(out, err)
        if iclip == 0:
            save_as_jpg = False
            main_status = status 
            if save_as_jpg:
                # this does not work, because PIL requires uint in [0, 255]
                # but interpolation in transforms introduces into float numbers.
                # see https://stackoverflow.com/a/55319979
                for i in range(num_step + 1):
                    fname = f"{name}.{flag}.{num_step + 1}_{i + 1:02}"
                    exist = os.path.isfile(f"{froot}/{fname}.jpg")
                    if not exist:
                        continue
                    image = T(PILImage.open(f"{froot}/{fname}.jpg")).numpy() 
                    image = np.moveaxis(image, 0, -1)
                    image = PILImage.fromarray(image)
                    image.save(f"{frame_np_root}/{fname}.jpg")
            else:
                arrays = {}
                images, index = [], []
                for i in range(num_step + 1):
                    fname = f"{name}.{flag}.{num_step + 1}_{i + 1:02}"
                    exist = os.path.isfile(f"{froot}/{fname}.jpg")
                    image = (
                        T(PILImage.open(f"{froot}/{fname}.jpg")).numpy() if exist else np.array([]) 
                    )
                    arrays[f"{i + 1:02}"] = image
                    if exist:
                        images.append(image)
                        index.append(i)
                np.savez_compressed(
                    f"{frame_np_root}/{name}.{flag}", **arrays
                )
            """
            images = torch.tensor(np.stack(images, axis=0))
            print(f"{threading.current_thread().ident} {images.shape} begins") 
            images = clip_model(images).numpy()
            print(f"{threading.current_thread().ident} {images.shape} done") 
            clip_out = {f"{i + 1:02}": np.array([]) for i in range(num_step + 1)}
            for i, real_i in enumerate(index):
                key = f"{real_i + 1:02}"
                clip_out[key] = images[i]
            np.savez_compressed(
                f"{frame_en_root}/{name}.{flag}", **clip_out
            )
            """
    return main_status

def rm_video(vfile):
    """
    :param vfile: str: the video file
    """
    if not os.path.isfile(vfile):
        return None, None, 1 
    return run(["rm", vfile])

def mp_worker(ytid):
    """
    :param ytid: (ytid, [(start_time, end_time, [label0, label1, ...])]) 
    """
    name = ytid[0] 
    try: # download
        #if name != "---1_cCGK4M" and name != "--PJHxphWEs":
        #    return name, 0 # debug 
        code, vfile = 1, f"{vroot}/{name}.mp4" 
        if not os.path.isfile(vfile):
            out, err, code, vfile = dl_video(ytid)
        #print(ytid, vfile)
    except Exception as e:
        code, vfile = 1, None
        print(f"Err in downloading {name}: {e}")
        pass
    flag = 1 if code == 0 else 0
    if vfile is None: 
        return name, flag 
    

    try: # clip extraction
        clips = collect_clip(ytid, vfile)
    except Exception as e:
        print(f"Err in clip extraction {name}: {e}")
        clips = []
        pass

    try: # frame extraction
        status = collect_frame(ytid, vfile, clips)
        assert isinstance(status, list)
        flag = int(all(status))
    except Exception as e:
        print(f"Err in frame extraction {name}: {e}")
        flag = 0
        pass
    
    try: # audio clip
        out, err, code = clip_audio(ytid, vfile, clips)
        flag = flag & (code == 0)
        #print(f"f{flag}", f"\n-->>>\n{out.strip()}\n{err.strip()}\n<<<--")
    except Exception as e:
        print(f"Err in audio clipping {name}: {e}")
        flag = 0
        pass

    try: # upload 
        #if flag == 1: # further take care
        upload(ytid)
    except Exception as e:
        print(f"Err in uploading video {name}: {e}")
        pass

    
    try: # video remove
        #if (not keepdata) and flag == 1: # further take care
        rm_video(vfile)
    except Exception as e:
        print(f"Err in deleting video {name}: {e}")
        pass
    return name, flag 

def mp_handler(param_list, nprocess=1, secs=30):
    """
    :param param_list: [ytid]
    :param nprocess: int
    :param secs: check pool status every #secs
    """
    num_task = len(param_list)
    print(f"total {num_task - len(done_set)} / {num_task} videos to download.")
    p = multiprocessing.Pool(nprocess)
    def write_err(results):
        with open(err_file, 'w') as f:
            for name, status in results: # 0: fail; 1: success
                #f.write(f"{name} {status}\n")
                if status == 0:
                    f.write(f"{name}\n")
    r = p.map_async(mp_worker, param_list, callback=write_err)
    if multiprocessing.current_process().name == 'MainProcess':
        k, c = 50, 0
        n = len(param_list)
        while not r.ready():
            c += 1 
            print(f"{r._number_left}", end=" ")
            if c % k == 0: print()
            time.sleep(secs)
    r.wait()
    p.close()
    p.join()

if __name__ == '__main__':
    csv_data = prepare(cfg, True)
    destroy(False)
    _, dict_ytids = collect_ytid(csv_data)
    #print(dict_ytids["--PJHxphWEs"])
    #import sys; sys.exit(0) 
    ytids = list(dict_ytids.items())[cfg.chunk_b : cfg.chunk_e]
    mp_handler(ytids, nprocess=cfg.nprocess, secs=cfg.peeprate)

