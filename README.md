# VIP-ANT: VIsually-Pivoted Audio and(N) Text

Code for the paper *[Connecting the Dots between Audio and Text without Parallel Data through Visual Knowledge Transfer](https://arxiv.org/abs/2112.08995)* @ NAACL 2022.

![VIP-ANT pivots audio and text via visual imagination.](https://drive.google.com/uc?id=13PnfOt4U6f86et-ebHs-nTWwMj0W6Ycv)

## Data

[AudioSet](https://research.google.com/audioset/) can be downloaded and preprocessed via this [tool](https://github.com/zhaoyanpeng/audioset-dl).

### AudioSet Data

See [AudioSet](https://github.com/zhaoyanpeng/audioset-dl/blob/beta/note/audioset.md). It elaborates on our customized index files for pre-training on AudioSet.

### Curated Audio-Text Data

See [AudioTxt](https://github.com/zhaoyanpeng/audioset-dl/blob/beta/note/audiotxt.md). It elaborates on our curation methods and customized index files for audio-text fine-tuning.

## Vision-Audio (VA) Pre-training

Check out the running script `bash/run_bimodal_va.sh`.

## Audio-Text (AT) Fine-tuning

Check out the running script `bash/run_bimodal_at.sh`. Fine-tuning starts with a VA pre-trained [audio encoder](https://storage.googleapis.com/ai2-mosaic-public/projects/vipant/model/01FFQTZK9YBPRDQHHR6157AGBR/00071478.pth).

## Pre-trained Models

We provide a checkpoint that performs best for each task.

### Pre-trained Models for Audio-Text Retrieval

| Model | AudioCaps | Clotho (18s) | Clotho (10s) |
|:-:|-:|-:|-:|
| VIP-ANT | [00051623](https://storage.googleapis.com/ai2-mosaic-public/projects/vipant/model/01FFQTZK9YBPRDQHHR6157AGBR/00051623.pth) | [00043681](https://storage.googleapis.com/ai2-mosaic-public/projects/vipant/model/01FFQTZK9YBPRDQHHR6157AGBR/00043681.pth) | [00043681](https://storage.googleapis.com/ai2-mosaic-public/projects/vipant/model/01FFQTZK9YBPRDQHHR6157AGBR/00043681.pth) |
| +AT w/ GC | [00006210](https://storage.googleapis.com/ai2-mosaic-public/projects/vipant/model/01FM2Y9HJ896B2G6NKRXTEVXZ7/00006210.pth) | [00006900](https://storage.googleapis.com/ai2-mosaic-public/projects/vipant/model/01FM2Y9HJ896B2G6NKRXTEVXZ7/00006900.pth) | [00004140](https://storage.googleapis.com/ai2-mosaic-public/projects/vipant/model/01FM2Y9HJ896B2G6NKRXTEVXZ7/00004140.pth) |

### Pre-trained Models for Zero-shot Audio Classification

| Model | ESC50 (w/ prompt) | US8K (w/ prompt) |
|:-:|-:|-:|
| VIP-ANT | [00083391](https://storage.googleapis.com/ai2-mosaic-public/projects/vipant/model/01FFQTZK9YBPRDQHHR6157AGBR/00083391.pth) | [00079420](https://storage.googleapis.com/ai2-mosaic-public/projects/vipant/model/01FFQTZK9YBPRDQHHR6157AGBR/00079420.pth) |
| +AT w/ GC | [00004140](https://storage.googleapis.com/ai2-mosaic-public/projects/vipant/model/01FM2Y9HJ896B2G6NKRXTEVXZ7/00004140.pth) | [00004140](https://storage.googleapis.com/ai2-mosaic-public/projects/vipant/model/01FM2Y9HJ896B2G6NKRXTEVXZ7/00004140.pth) |

## Dependencies

`Dockerfile` defines minimum dependencies of the repo.

## License
MIT
