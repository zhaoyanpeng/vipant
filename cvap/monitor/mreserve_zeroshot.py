from omegaconf import OmegaConf
import os, re
import warnings
from typing import Union, List
from collections import defaultdict

import copy
import time
import torch
import numpy as np
from torch import nn
from PIL import Image
from tqdm import tqdm

import jax.numpy as jnp
import torch.distributed as dist
from torch.nn.parallel import data_parallel
from torch.nn.parallel import DistributedDataParallel

from ..model import extract_model_file
from ..module import LARS, exclude_bias_or_norm, adjust_learning_rate
from ..dataset.audio import build_dataloader_list

from .finetune import Monitor

class Monitor(Monitor):
    def __init__(self, cfg, echo, device):
        super(Monitor, self).__init__(cfg, echo, device)

    def build_data(self):
        self.loader_list, self.lid2str, self.lid2int, self.label_map = build_dataloader_list(self.cfg, mreserve=True)
        return len(self.lid2str)
        
    def learn(self):
        if not self.model.training:
            if self.cfg.running.zero_shot:
                self.echo("(Zero-shot) Evaluating started...")
                if self.cfg.model_file.endswith(".out"):
                    with torch.no_grad():
                        self.repeated_zero_shot() # multiple evaluations
                else:
                    with torch.no_grad():
                        report = self.standard_zero_shot(samples=self.cfg.running.eval_samples)
                    self.echo(f"{report}")
                return None
            self.echo("Evaluating started...")
            with torch.no_grad():
                report = self.infer(self.dataloader, samples=self.cfg.running.eval_samples)
                self.echo(f"{report}")
                return None 
        else:
            raise ValueError(f"Learning is not supported right now.")

    def make_batch(self, batch):
        batch = (
            jnp.array(batch[0]), # audio (dummy and ignored)
            jnp.array(batch[1]), # label
            batch[2], # label name
            batch[3], # video segments
        )
        return batch # bare tensors

    def standard_zero_shot(self, samples=float("inf"), iepoch=0):
        report_by_fold = list()
        device_ids = [i for i in range(self.cfg.num_gpus)]
        
        for ifold, (_, evalloader_fn) in enumerate(self.loader_list):
            _, dataloader = evalloader_fn()
            nsample, nchunk, nbatch = 0, 1, len(dataloader)

            if isinstance(self.model, DistributedDataParallel):
                dataloader.sampler.set_epoch(iepoch)
                nchunk = self.cfg.num_gpus
            
            break_me = False

            start_time = time.time()
            for ibatch, batch in enumerate(dataloader):
                audios, labels, names, videos = self.make_batch(batch)
                self.model(audios, labels, device_ids=device_ids, videos=videos[0], names=names)
                #self.echo(f"{audios.shape} {labels.shape}")
                #print(audios, labels, names, videos)
                #import sys; sys.exit(0)
                if ibatch % self.cfg.running.peep_rate == 0:
                    self.echo(ibatch)
                nsample += audios.shape[0] * nchunk
                if nsample >= samples:
                    break_me = True
                    break
            self.echo(f"{ifold:>2}th fold: # sample {nsample}; {nsample / (time.time() - start_time):.2f} samples/s")
            
            if break_me:
                break

        prompt = self.cfg.running.prompt.strip()
        prompt = "" if prompt == "" else prompt + " "
        lid2str = [prompt + self.lid2str[i].replace("_", " ") for i in range(len(self.lid2str))]
        text_features = self.model.encode_text(lid2str, device_ids=device_ids, label_map=self.label_map) #None) #
        self.echo(lid2str)

        model = self.model.module if isinstance(self.model, DistributedDataParallel) else self.model
        report = model.report(text=text_features, label_map=None) #self.label_map)
        self.echo(f"{report}")

        precision = re.search("=\s(\d+\.\d+)\s\@", report)
        assert precision is not None, f"invalid report: `{report}`"
        precision = float(precision.group(1))
        report_by_fold.append(precision)
        return f"{precision:2.2f} for zero-shot classification."

