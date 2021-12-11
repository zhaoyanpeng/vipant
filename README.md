# VIP-ANT: VIsually-Pivoted Audio and(N) Text

Code for the paper *[Connecting the Dots between Audio and Text without Parallel Data through Visual Knowledge Transfer](https://arxiv.org/abs/2112.08995)*.

![VIP-ANT pivots audio and text via visual imagination.](https://drive.google.com/uc?id=13PnfOt4U6f86et-ebHs-nTWwMj0W6Ycv)

## Data

[AudioSet](https://research.google.com/audioset/) can be downloaded and preprocessed via this [tool](https://github.com/zhaoyanpeng/audioset-dl).

## Vision-Audio (VA) Pre-training

Check out the running script `bash/run_bimodal_va.sh`.

## Audio-Text (AT) Fine-tuning

Check out the running script `bash/run_bimodal_at.sh`.

## Dependencies

`Dockerfile` defines the minimum dependencies of the repo.

## Citing VIP-ANT
```
@misc{vip-ant,
      title={Connecting the Dots between Audio and Text without Parallel Data through Visual Knowledge Transfer},
      author={Yanpeng Zhao and Jack Hessel and Youngjae Yu and Ximing Lu and Rowan Zellers and Yejin Choi},
      url={https://arxiv.org/abs/2112.08995},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      eprint={2112.08995},
      year={2021},
}
```

## License
MIT
