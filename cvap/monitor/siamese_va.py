from omegaconf import OmegaConf
import os, re
from collections import defaultdict

import time
import torch
import numpy as np
from torch import nn

import tensorflow as tf
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import data_parallel
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from .cvap import Monitor
from ..util import numel
from ..model import build_main_model
from ..module import LARS, exclude_bias_or_norm, adjust_learning_rate
from ..dataset import build_dataloader 

class Monitor(Monitor):
    def __init__(self, cfg, echo, device):
        super(Monitor, self).__init__(cfg, echo, device)

    def make_batch(self, batch):
        def scale_images(images):
            if images.dim() != 2 and images.shape[-1] != self.cfg.running.resolution and \
                list(images.shape[1:]) != [1, 1, 1]:
                images = F.interpolate(
                    images,
                    self.cfg.running.resolution,
                    mode="bilinear",
                    align_corners=False,
                )
            return images
        #print(batch[0].shape, batch[1].shape, batch[2].shape, batch[3].shape, batch[4].shape)
        images = (torch.tensor(batch[0], device=self.device)
            if self.cfg.model.loss.vp or self.cfg.model.loss.ap else None
        )
        images_v1 = scale_images(
            torch.tensor(batch[1], device=self.device) # (c, h, w)
        )
        images_v2 = scale_images(
            torch.tensor(batch[2], device=self.device) # (c, h, w)
        ) if self.cfg.model.loss.vv else None
        audios_v1 = torch.tensor(batch[3], device=self.device)
        audios_v2 = (torch.tensor(batch[4], device=self.device)
            if self.cfg.model.loss.aa else None
        )
        """
        print(
            images.shape if images is not None else None,
            self.cfg.running.resolution,
            images_v1.shape,
            images_v2.shape if images_v2 is not None else None,
            audios_v1.shape,
            audios_v2.shape if audios_v2 is not None else None,
        )
        """
        #import sys; sys.exit(0)
        batch = (
            images, images_v1, images_v2, audios_v1, audios_v2, batch[-1], # sample id or name
        )
        return batch # bare tensors

    def epoch(self, iepoch):
        all_time = defaultdict(list)
        self.timeit(all_time)        
        device_ids = [i for i in range(self.cfg.num_gpus)]
        nchunk = dist.get_world_size() if torch.distributed.is_initialized() else 1  
        warmup_step_rate = max(self.cfg.optimizer.warmup_steps // 20, 1)
        for step, batch in enumerate(self.dataloader, start=iepoch * len(self.dataloader)):
            images, images_v1, images_v2, audios_v1, audios_v2, _ = self.make_batch(batch)
            self.timeit(all_time, key="data")

            if self.cfg.optimizer.use_lars:
                adjust_learning_rate(self.cfg.optimizer, self.optimizer, self.dataloader, step)

            inc = 0
            force_eval = False # recommended by SGDR
            warmup = not self.cfg.optimizer.use_lars and self.cfg.optimizer.warmup and \
                (self.total_step + inc) <= self.cfg.optimizer.warmup_steps
            # it is important to always warm up lr at the first step otherwise
            # the optimizer will use the default / initial lr
            if warmup and ((self.total_step + inc) % warmup_step_rate == 0 or self.total_step == 0):
                ratio = ((self.total_step + inc) / self.cfg.optimizer.warmup_steps) # * self.cfg.optimizer.lr
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = ratio * param_group["initial_lr"]
                lrs = [param_group['lr'] for param_group in self.optimizer.param_groups]
                force_eval = lrs == self.scheduler.base_lrs
                lrs = [f"{lr:.2e}" for lr in lrs]
                self.echo(f"warmup lr: {' '.join(lrs)} @ {self.total_step}")

            self.optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                loss = self.model(
                    images, images_v1, audios_v1, device_ids=device_ids,
                    text_v1=None, images_v2=images_v2, audios_v2=audios_v2
                )
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if not self.cfg.optimizer.use_lars and self.cfg.optimizer.batch_sch and not warmup:
                old_lrs = " ".join([f"{x:.2e}" for x in self.scheduler.get_last_lr()])
                self.scheduler.step() # after all warmup is completed
                if isinstance(self.scheduler, (CosineAnnealingWarmRestarts,)):
                    force_eval = self.scheduler.get_last_lr() == self.scheduler.base_lrs
                #self.echo(f"do step lr {old_lrs}")

            self.timeit(all_time, key="model")

            if False and self.cfg.rank == 0:
                print(f"doing some check on unused params... {dist.get_world_size()}")
                for k, v in self.model.named_parameters():
                    if v.requires_grad and v.grad is None:
                        print(f"--> {k}")

            self.total_step += 1
            self.total_loss += loss.detach()
            self.total_inst += images.shape[0] * nchunk if images is not None else audios_v1.shape[0] * nchunk
            if force_eval or (self.cfg.rank == 0 and self.total_step % self.cfg.running.peep_rate == 0):
                def grad_norm():
                    return sum(
                        [p.grad.norm(p=2) ** 2 for p in self.params if p.grad is not None]
                    ).item() ** 0.5
                lr_w = self.optimizer.param_groups[0]['lr']
                lr_b = self.optimizer.param_groups[1]['lr']
                msg = self.model.report(**{"nstep": self.total_step})
                self.echo(
                    f"epoch {iepoch:>4} step {self.total_step}\t" + #gnorm {grad_norm():.2f} " +
                    f"lr_w {lr_w:.2e} lr_b {lr_b:.2e} loss {self.total_loss / self.total_step:.3f} " + 
                    f"{msg} {self.total_inst / (time.time() - self.start_time):.2f} samples/s" 
                )
            if force_eval or self.total_step % self.cfg.running.save_rate == 0 or (
                    self.cfg.running.save_epoch and self.total_step % len(self.dataloader) == 0
                ): # distributed eval
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

        if not self.cfg.optimizer.use_lars and not self.cfg.optimizer.batch_sch:
            self.scheduler.step()
        self.timeit(all_time, show=True)
        
    def infer(self, dataloader, samples=float("inf"), iepoch=0):
        losses, nsample, nchunk, nbatch = 0, 0, 1, len(dataloader)
        device_ids = [i for i in range(self.cfg.num_gpus)]
        if isinstance(self.model, DistributedDataParallel):
            dataloader.sampler.set_epoch(iepoch)
            nchunk = self.cfg.num_gpus
        peep_rate = max(10, (len(dataloader) // 10))
        start_time = time.time()
        for ibatch, batch in enumerate(dataloader):
            if nsample >= samples:
                #print(f"{nsample}\t{ibatch}/{nbatch} continue")
                break #continue # iterate through every batch 
            images, images_v1, _, audios_v1, _, names = self.make_batch(batch)
            #msg = f"{images[0, 0, 50, 50:55]} {audios[0, 0, 50, 50:55]}" # if ibatch == 0 else ""
            #print(f"{nsample}\t{ibatch}/{nbatch} done {msg}")
            loss = self.model(images, images_v1, audios_v1, device_ids=device_ids, names=names)
            nsample += images.shape[0] * nchunk if images is not None else audios_v1.shape[0] * nchunk
            losses += loss or 0.
            if self.cfg.rank == 0 and (ibatch + 1) % peep_rate == 0:
                self.echo(
                    f"step {ibatch}\t" + #gnorm {grad_norm():.2f} " +
                    f"loss {losses / (ibatch + 1):.8f} " +
                    f"{nsample / (time.time() - start_time):.2f} samples/s"
                )
        model = self.model.module if isinstance(self.model, DistributedDataParallel) else self.model
        self.echo(f"# sample {nsample}; {nsample / (time.time() - start_time):.2f} samples/s")
        return model.report(gold_file=self.gold_file)
