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

import torch.distributed as dist
from torch.nn.parallel import data_parallel
from torch.nn.parallel import DistributedDataParallel

from ..model import AudioClassifier as Model
from ..module import LARS, exclude_bias_or_norm, adjust_learning_rate
from ..dataset.audio import build_dataloader_list

class Monitor(object):
    def __init__(self, cfg, echo, device):
        super(Monitor, self).__init__()
        self.cfg = cfg
        self.echo = echo
        self.device = device
        output_dim = self.build_data()
        model = Model(cfg, echo)
        tunable_params = model.build(**{"output_dim": output_dim})
        self.model = DistributedDataParallel(
            model, device_ids=[cfg.rank], find_unused_parameters=True
        ) if torch.distributed.is_initialized() else model 
        self.model.train(not cfg.eval)
        self.build_optimizer(tunable_params)

    def reinitialize(self, cfg, echo):
        self.echo("Reinitialize everything except `dataloader_list`.")
        model = Model(cfg, echo)
        output_dim = len(self.lid2str)
        tunable_params = model.build(**{"output_dim": output_dim})
        self.model = DistributedDataParallel(
            model, device_ids=[cfg.rank], find_unused_parameters=True
        ) if torch.distributed.is_initialized() else model 
        self.model.train(not cfg.eval)
        self.build_optimizer(tunable_params)

    def build_data(self):
        self.loader_list, self.lid2str = build_dataloader_list(self.cfg)
        return len(self.lid2str)
        
    def learn(self):
        if not self.model.training:
            self.echo("Evaluating started...")
            with torch.no_grad():
                report = self.infer(self.dataloader, samples=self.cfg.running.eval_samples)
                self.echo(f"{report}")
                return None 
        #self.save() 
        report_by_fold = list() 
        for ifold, (dataloader_fn, evalloader_fn) in enumerate(self.loader_list):
            _, self.dataloader = dataloader_fn()
            _, self.evalloader = evalloader_fn() 

            self.echo(f"Training started ({ifold})...")
            self.last_time = 0.
            self.total_loss = 0
            self.total_step = 0
            self.total_inst = 0
            self.start_time = time.time()
            self.scaler = torch.cuda.amp.GradScaler()
            self.report_by_epoch = list()

            for iepoch in range(self.cfg.optimizer.epochs):
                if isinstance(self.model, DistributedDataParallel):
                    self.dataloader.sampler.set_epoch(iepoch)
                if iepoch >= 1:
                    pass #break
                self.epoch(iepoch)
            #break
            report_by_fold.append(self.report_by_epoch)
            if ifold == 1:
                pass #break
            self.reinitialize(self.cfg, self.echo)
        self.summary_report(report_by_fold)

    def summary_report(self, report):
        report = np.array(report)
        self.echo(f"\n{report}")
        nfold, nepoch = report.shape[:2]
        self.echo(f"Total {nepoch} epochs for each of {nfold} folds.")

        report_sum = report.sum(0)
        best_epoch = report_sum.argmax()
        best_precisions = report[:, best_epoch]
        mean, std = best_precisions.mean(), best_precisions.std()
        self.echo(f"Best mean and std: {mean:2.2f} \\pm {std:2.2f} in the {best_epoch}th epoch.")

        max_epoch = report.argmax(1)
        max_precisions = report.max(1) #
        #_, max_epoch = np.where(report == max_precisions[..., None])
        mean, std = max_precisions.mean(), max_precisions.std()
        self.echo(f"Max mean and std: {mean:2.2f} \\pm {std:2.2f} in the {max_epoch}th epoch.")

    def make_batch(self, batch):
        batch = (
            torch.tensor(
                batch[0], device=self.device
            ).unsqueeze(1), # audio 
            torch.tensor(
                batch[1], device=self.device
            ), # label
            batch[2], # label name
        )
        return batch # bare tensors

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
            audios, labels, _ = self.make_batch(batch)
            self.timeit(all_time, key="data")

            if self.cfg.optimizer.use_lars:
                adjust_learning_rate(self.cfg.optimizer, self.optimizer, self.dataloader, step)
            
            self.optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                loss = self.model(audios, labels, device_ids=device_ids)
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
            self.total_inst += audios.shape[0] * nchunk
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
            if self.total_step % self.cfg.running.save_rate == 0 or (
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
                    precision = re.search("=\s(\d+\.\d+)\s\@", report)
                    assert precision is not None, f"invalid report: `{report}`"
                    precision = float(precision.group(1))
                    self.report_by_epoch.append(precision)
                    self.echo(f"{report}")
                if self.cfg.rank == 0:
                    pass #self.save()
            self.timeit(all_time, key="report")

        if not self.cfg.optimizer.use_lars:
            self.scheduler.step()
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
            audios, labels, names = self.make_batch(batch)
            #msg = f"{audios[0, 0, 50, 50:55]} {audios[0, 0, 50, 50:55]}" # if ibatch == 0 else ""
            #print(f"{nsample}\t{ibatch}/{nbatch} done {msg}")
            loss = self.model(audios, labels, device_ids=device_ids, names=names)
            nsample += audios.shape[0] * nchunk
        model = self.model.module if isinstance(self.model, DistributedDataParallel) else self.model
        self.echo(f"# sample {nsample}; {nsample / (time.time() - start_time):.2f} samples/s")
        return model.report()

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
        if self.cfg.optimizer.use_lars:
            self.optimizer = LARS(
                param_groups, 
                lr=0., 
                weight_decay=self.cfg.optimizer.weight_decay,
                weight_decay_filter=exclude_bias_or_norm,
                lars_adaptation_filter=exclude_bias_or_norm,
            )
        else:
            self.optimizer = torch.optim.Adam(
                param_groups, self.cfg.optimizer.lr, weight_decay=5e-7, betas=(0.95, 0.999)
            )
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer, list(range(5, 26)), gamma=0.85
            )
        debug = False 
        if not debug:
            return
        self.echo(f"Gradienting The Following Parameters:")
        for k, v in self.model.named_parameters():
            if v.requires_grad:
                self.echo(f"{k} {v.size()}")

