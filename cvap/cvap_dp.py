from omegaconf import OmegaConf
import os, re
import warnings
from typing import Union, List
from collections import defaultdict

import time
import torch
import numpy as np
from torch import nn
from PIL import Image
from tqdm import tqdm

import torch.distributed as dist
from torch.nn.parallel import data_parallel
from torch.nn.parallel import DistributedDataParallel

from .text_head import build_text_head
from .loss_head import build_loss_head
from .image_head import build_image_head
from .audio_head import build_audio_head

from .datasets import ImageAudioCollator, ImageAudioDataset, ImageAudioDatasetSrc, ImageAudioDatasetNpz
from .utils import exclude_bias_or_norm, adjust_learning_rate
from .module import LARS

from clip import load 


class CVAP(nn.Module):
    def __init__(self, cfg, echo):
        super(CVAP, self).__init__()
        self.cfg = cfg
        self.echo = echo

    def forward(self, images, audios, *args, **kwargs):
        device_ids = kwargs.get("device_ids", [0])
        # how to asynchronize the two `data_parallel` 
        kwargs = {"normalized": True, "names": kwargs.get("names", None)}
        image_features = data_parallel(
            self.image_head, images, device_ids=device_ids, module_kwargs=kwargs
        )
        audio_features = data_parallel(
            self.audio_head, audios, device_ids=device_ids, module_kwargs=kwargs
        )
        loss = self.loss_head(image_features, audio_features, **kwargs)
        return loss     

    def collect_audio_state_dict(self):
        return (
            self.audio_head.state_dict(), 
            self.loss_head.state_dict(),
        )

    def collect_state_dict(self):
        return (
            self.image_head.state_dict(), 
            self.audio_head.state_dict(), 
            self.loss_head.state_dict(),
        )

    def report(self, gold_file=None):
        if not dist.is_initialized() or dist.get_rank() == 0:
            return self.loss_head.report(gold_file=gold_file)
        else:
            return ""
    
    def build(self):
        tunable_params = dict()
        if self.cfg.eval:
            model_file = f"{self.cfg.model_root}/{self.cfg.model_name}/{self.cfg.model_file}"
            self.echo(f"Loading from {model_file}")
            checkpoint = torch.load(model_file, map_location="cpu")
            local_cfg = checkpoint["cfg"]
            local_str = OmegaConf.to_yaml(local_cfg)
            self.echo(f"Old configs:\n\n{local_str}")
            audio_head_sd, loss_head_sd = checkpoint["model"]

            rcfg = local_cfg.running
            model, self.T = load(
                rcfg.clip_model_name, rcfg.clip_model_root, device="cpu", jit=False
            )
            image_head_sd = model.visual.state_dict()
            self.image_head = build_image_head(local_cfg.model.image)
            self.image_head.copy_state_dict(image_head_sd)

            self.audio_head = build_audio_head(local_cfg.model.audio)
            self.audio_head.load_state_dict(audio_head_sd)

            self.loss_head = build_loss_head(local_cfg.model.loss)
            self.loss_head.load_state_dict(loss_head_sd)
            self.cuda(self.cfg.rank) 
        else:
            rcfg = self.cfg.running
            model, self.T = load(
                rcfg.clip_model_name, rcfg.clip_model_root, device="cpu", jit=False
            )
            image_head_sd = model.visual.state_dict()
            self.image_head = build_image_head(self.cfg.model.image)
            self.image_head.copy_state_dict(image_head_sd)

            self.audio_head = build_audio_head(self.cfg.model.audio)
            self.audio_head.copy_state_dict(image_head_sd)
                
            extra_sd = {"logit_scale": model.logit_scale}
            self.loss_head = build_loss_head(self.cfg.model.loss)
            self.loss_head.copy_state_dict(extra_sd)

            tunable_params = {
                f"audio_head.{k}": v for k, v in self.audio_head.named_parameters()
            } 
            tunable_params.update({
                f"loss_head.{k}": v for k, v in self.loss_head.named_parameters()
            })
            self.cuda(self.cfg.rank)
        return tunable_params


class Monitor(object):
    def __init__(self, cfg, echo, device):
        super(Monitor, self).__init__()
        self.cfg = cfg
        self.echo = echo
        self.device = device
        self.build_data()
        model = CVAP(cfg, echo)
        tunable_params = model.build()
        self.model = DistributedDataParallel(
            model, device_ids=[cfg.rank], find_unused_parameters=True
        ) if torch.distributed.is_initialized() else model 
        self.model.train(not cfg.eval)
        self.build_optimizer(tunable_params)

    def build_data(self):
        def build_dataloader(cfg, data_name, shuffle=True, train=True):
            ddp_mode = torch.distributed.is_initialized()
            rcfg = cfg.running
            if data_name.startswith("src"):
                dataset = ImageAudioDatasetSrc(rcfg, data_name, train)
            elif data_name.startswith("npz"):
                dataset = ImageAudioDatasetNpz(rcfg, data_name, train)
            else:
                dataset = ImageAudioDataset(rcfg, data_name, train)
            if ddp_mode:
                assert self.cfg.optimizer.batch_size % self.cfg.num_gpus == 0
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
        rcfg = self.cfg.running
        data_name = rcfg.eval_name if self.cfg.eval else rcfg.data_name
        _, self.dataloader = build_dataloader(
            self.cfg, data_name, shuffle=(not self.cfg.eval), train=(not self.cfg.eval)
        )
        self.echo(f"Instantiating main dataloader from `{data_name}': total {len(self.dataloader)} batches.")
        self.gold_file = f"{rcfg.data_root}/{data_name}.csv"
        # evaluation
        eval_name = "IGNORE_ME" if self.cfg.eval else rcfg.eval_name
        data_path = f"{rcfg.data_root}/{eval_name}"
        _, self.evalloader = build_dataloader(
            self.cfg, eval_name, shuffle=False, train=False
        ) if os.path.isdir(data_path) or os.path.isfile(f"{data_path}.csv") else (None, None)
        if self.evalloader is not None:
            self.echo(f"Will do evaluation every {rcfg.save_rate} steps on {len(self.evalloader)} batches.")
            self.gold_file = f"{rcfg.data_root}/{eval_name}.csv"

    def learn(self):
        if not self.model.training:
            self.echo("Evaluating started...")
            with torch.no_grad():
                report = self.infer(self.dataloader, samples=self.cfg.running.eval_samples)
                self.echo(f"{report}")
                return None 
        self.echo("Training started...")
        self.last_time = 0.
        self.total_loss = 0
        self.total_step = 0
        self.total_inst = 0
        self.start_time = time.time()
        self.scaler = torch.cuda.amp.GradScaler()
        #self.save() 
        for iepoch in range(self.cfg.optimizer.epochs):
            if isinstance(self.model, DistributedDataParallel):
                self.dataloader.sampler.set_epoch(iepoch)
            if iepoch >= 1:
                pass #break
            self.epoch(iepoch)

    def make_batch(self, batch):
        batch = (
            torch.tensor(
                batch[0], device=self.device
            ), 
            torch.tensor(
                batch[1], device=self.device
            ).unsqueeze(1),
            batch[2], # sample id or name
        )
        return batch # bare tensors

        images, audios = batch["image"], batch["audio"]

        images = images.cuda(self.cfg.rank, non_blocking=True)
        audios = audios.cuda(self.cfg.rank, non_blocking=True)
        return images, audios # directly return dict of tensors

        images = torch.tensor(
            np.concatenate(images, axis=0), device=self.device
        ) #.cuda(self.cfg.rank, non_blocking=True)
        audios = torch.tensor(
            np.concatenate(audios, axis=0), device=self.device
        ).unsqueeze(1) #.cuda(self.cfg.rank, non_blocking=True)
        #print(f"make_batch {dist.get_rank()} {batch['name']}")
        return images, audios

    def timeit(self, time_dict, key=None, show=False):
        if self.cfg.rank != 0:
            return 
        if show: # print
            report = ""
            for k, v in time_dict.items():
                report += f"{k} {np.mean(v):.2f} "
            self.echo(f"Time (s): {report.strip()}; # step {self.total_step} # sample {self.total_inst}")
            return
        if key is None: # initialize
            self.last_time = time.time()
        else: # update
            this_time = time.time()
            time_dict[key].append(this_time - self.last_time)
            self.last_time = this_time

    def epoch(self, iepoch):
        all_time = defaultdict(list)
        self.timeit(all_time)        
        device_ids = [i for i in range(self.cfg.num_gpus)]
        nchunk = dist.get_world_size() if torch.distributed.is_initialized() else 1  
        for step, batch in enumerate(self.dataloader, start=iepoch * len(self.dataloader)):
            images, audios, _ = self.make_batch(batch)
            self.timeit(all_time, key="data")

            #print(images.size(), audios.size())
            #break

            adjust_learning_rate(self.cfg.optimizer, self.optimizer, self.dataloader, step)

            self.optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                loss = self.model(images, audios, device_ids=device_ids)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            self.timeit(all_time, key="model")

            if False and self.cfg.rank == 0:
                print(f"doing some check on unused params... {dist.get_world_size()}")
                for k, v in self.model.named_parameters():
                    if v.requires_grad and v.grad is None:
                        print(f"--> {k}")

            self.total_step += 1
            self.total_loss += loss.detach()
            self.total_inst += images.shape[0] * nchunk
            if self.cfg.rank == 0 and self.total_step % self.cfg.running.peep_rate == 0:
                def grad_norm():
                    return sum(
                        [p.grad.norm(p=2) ** 2 for p in self.params if p.grad is not None]
                    ).item() ** 0.5
                lr_w = self.optimizer.param_groups[0]['lr']
                lr_b = self.optimizer.param_groups[1]['lr']
                self.echo(
                    f"epoch {iepoch:>4} step {self.total_step}\t" + #gnorm {grad_norm():.2f} " +
                    f"lr_w {lr_w:.2e} lr_b {lr_b:.2e} loss {self.total_loss / self.total_step:.3f} " + 
                    f"{self.total_inst / (time.time() - self.start_time):.2f} samples/s" 
                )
            if self.total_step % self.cfg.running.save_rate == 0: # distributed eval
                report = ""
                if self.evalloader is not None:
                    self.model.train(False)
                    with torch.no_grad():
                        report = self.infer(
                            self.evalloader, samples=self.cfg.running.eval_samples, iepoch=iepoch
                        )
                    self.model.train(True)
                if report != "":
                    self.echo(f"{report}")
                if self.cfg.rank == 0:
                    self.save()
            self.timeit(all_time, key="report")
        self.timeit(all_time, show=True)
        
    def infer(self, dataloader, samples=float("inf"), iepoch=0):
        nsample, nchunk, nbatch = 0, 1, len(dataloader) 
        device_ids = [i for i in range(self.cfg.num_gpus)]
        if isinstance(self.model, DistributedDataParallel):
            dataloader.sampler.set_epoch(iepoch)
            nchunk = self.cfg.num_gpus
        start_time = time.time()
        for ibatch, batch in enumerate(dataloader):
            if nsample >= samples:
                #print(f"{nsample}\t{ibatch}/{nbatch} continue")
                break #continue # iterate through every batch 
            images, audios, names = self.make_batch(batch)
            #msg = f"{images[0, 0, 50, 50:55]} {audios[0, 0, 50, 50:55]}" # if ibatch == 0 else ""
            #print(f"{nsample}\t{ibatch}/{nbatch} done {msg}")
            loss = self.model(images, audios, device_ids=device_ids, names=names)
            nsample += images.shape[0] * nchunk
        model = self.model.module if isinstance(self.model, DistributedDataParallel) else self.model
        self.echo(f"# sample {nsample}; {nsample / (time.time() - start_time):.2f} samples/s")
        return model.report(gold_file=self.gold_file)

    def save(self):
        fsave = f"{self.cfg.model_root}/{self.cfg.model_name}/{self.total_step:08d}.pth"
        self.echo(f"Saving the checkpoint to {fsave}")
        model = self.model.module if isinstance(self.model, DistributedDataParallel) else self.model
        checkpoint = {
            "cfg": self.cfg, "model": model.collect_audio_state_dict(), # model.collect_state_dict(),
        }
        torch.save(checkpoint, fsave)

    def build_optimizer(self, tunable_params={}):
        if not self.model.training:
            return
        self.params = (
            list(tunable_params.values())
        )
        for k, v in tunable_params.items():
            if self.cfg.rank == 0:
                pass #self.echo(f"{k} {v.size()}")
        ddp = isinstance(self.model, DistributedDataParallel)
        for k, v in self.model.named_parameters():
            k = re.sub("^module\.", "", k) if ddp else k
            if f"{k}" not in tunable_params:
                v.requires_grad = False
        param_groups = [
            {"params": [p for p in self.params if p.ndim > 1]},
            {"params": [p for p in self.params if p.ndim < 2]},
        ]
        self.optimizer = LARS(
            param_groups, 
            lr=0., 
            weight_decay=self.cfg.optimizer.weight_decay,
            weight_decay_filter=exclude_bias_or_norm,
            lars_adaptation_filter=exclude_bias_or_norm,
        )
        debug = False 
        if not debug:
            return
        self.echo(f"Gradienting The Following Parameters:")
        for k, v in self.model.named_parameters():
            if v.requires_grad:
                self.echo(f"{k} {v.size()}")

