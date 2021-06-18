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
from torch.nn.parallel import DistributedDataParallel

from .text_head import build_text_head
from .loss_head import build_loss_head
from .image_head import build_image_head
from .audio_head import build_audio_head

from .datasets import ImageAudioCollator, ImageAudioDataset
from .utils import exclude_bias_or_norm, adjust_learning_rate
from .module import LARS

from clip import load 


class CVAP(nn.Module):
    def __init__(self, cfg, echo):
        super(CVAP, self).__init__()
        self.cfg = cfg
        self.echo = echo

    def forward(self, images, audios, *args, **kwargs):
        # use gather or reduce, that depends on the loss_head
        image_features = self.image_head(images)
        audio_features = self.audio_head(audios)
        
        if not self.loss_head.reduce:
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            audio_features = audio_features / audio_features.norm(dim=-1, keepdim=True)

            image_list = [torch.zeros_like(image_features) for _ in range(self.cfg.num_gpus)]
            audio_list = [torch.zeros_like(audio_features) for _ in range(self.cfg.num_gpus)]

            dist.all_gather(tensor_list=image_list, tensor=image_features.contiguous())
            dist.all_gather(tensor_list=audio_list, tensor=audio_features.contiguous())

            image_list[dist.get_rank()] = image_features 
            audio_list[dist.get_rank()] = audio_features 

            image_features = torch.cat(image_list)
            audio_features = torch.cat(audio_list)

            loss = self.loss_head(image_features, audio_features, normalized=True)
        else: # TODO barlow loss
            pass
        return loss     

    def collect_state_dict(self):
        return (
            self.image_head.state_dict(), self.audio_head.state_dict(), self.loss_head.state_dict(),
        )
    
    def build(self):
        tunable_params = dict()
        if self.cfg.eval:
            self.echo(f"Loading from {self.cfg.model_file}")
            checkpoint = torch.load(self.cfg.model_file, map_location="cpu")
            local_cfg = checkpoint["cfg"]
            self.echo(f"Old configs:\n{local_cfg}")
            image_head_sd, audio_head_sd = checkpoint["model"]
            self.image_head = build_image_head(local_cfg.model.image)
            self.image_head.load_state_dict(image_head_sd)
            self.audio_head = build_audio_head(local_cfg.model.audio)
            self.audio_head.load_state_dict(audio_head_sd)
            self.loss_head = build_loss_head(local_cfg.model.loss)
            self.loss_head.load_state_dict(loss_head_sd)
            self.eval()
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
        ) 
        self.build_optimizer(tunable_params)

    def build_data(self):
        assert self.cfg.optimizer.batch_size % self.cfg.num_gpus == 0
        def build_dataloader(cfg, data_name):
            rcfg = cfg.running
            dataset = ImageAudioDataset(rcfg, data_name)
            sampler = torch.utils.data.distributed.DistributedSampler(dataset) 
            per_device_batch_size = cfg.optimizer.batch_size // cfg.num_gpus
            dataloader = torch.utils.data.DataLoader(
                dataset, 
                batch_size=per_device_batch_size,
                collate_fn=ImageAudioCollator(),
                num_workers=cfg.num_proc,
                pin_memory=True,
                sampler=sampler,
                drop_last=True,
            )
            return sampler, dataloader
        pp = self.cfg.optimizer.batch_size // self.cfg.num_gpus
        rcfg = self.cfg.running
        _, self.dataloader = build_dataloader(self.cfg, rcfg.data_name)
        return
        # evaluation
        data_path = f"{rcfg.data_root}/{rcfg.eval_name}"
        _, self.evalloader = build_dataloader(
            self.cfg, rcfg.eval_name
        ) if os.path.isdir(data_path) else None, None
        if self.evalloader is not None:
            self.echo(f"Will do evaluation every {rcfg.peep_rate} steps.")

    def learn(self):
        if not self.model.training:
            with torch.no_grad():
                return self.infer(self.dataloader)
        self.echo("Training started...")
        self.total_loss = 0
        self.total_step = 0
        self.total_inst = 0
        self.start_time = time.time()
        self.scaler = torch.cuda.amp.GradScaler()
        #self.save() 
        for iepoch in range(self.cfg.optimizer.epochs):
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
            ).unsqueeze(1)
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

    def epoch(self, iepoch):
        if self.cfg.rank == 0:
            print(f"A {iepoch} x {len(self.dataloader)}")
            all_time = defaultdict(list)
            last_time = time.time()
        for step, batch in enumerate(self.dataloader, start=iepoch * len(self.dataloader)):
            images, audios = self.make_batch(batch)
            
            if self.cfg.rank == 0:
                #print(f"B {step} x {iepoch}")
                this_time = time.time()
                all_time["dataloader"].append(this_time - last_time)
                last_time = this_time
            #continue # test dataloader efficiency

            adjust_learning_rate(self.cfg.optimizer, self.optimizer, self.dataloader, step)

            self.optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                loss = self.model(images, audios) 
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if False and self.cfg.rank == 0:
                print(f"doing some check on unused params... {dist.get_world_size()}")
                for k, v in self.model.named_parameters():
                    if v.requires_grad and v.grad is None:
                        print(f"--> {k}")
            
            if self.cfg.rank == 0:
                #print(f"C")
                this_time = time.time()
                all_time["forward"].append(this_time - last_time)
                last_time = this_time

            self.total_step += 1
            self.total_loss += loss.detach()
            self.total_inst += images.shape[0] * dist.get_world_size() 
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
            if self.cfg.rank == 0 and self.total_step % self.cfg.running.save_rate == 0:
                report = ""
                if False and self.evalloader is not None:
                    self.eval()
                    with torch.no_grad():
                        report = self.infer(self.evalloader, 5000)
                    self.train()
                self.echo(f"{report}")
                self.save()
            
            if self.cfg.rank == 0:
                #print(f"D")
                this_time = time.time()
                all_time["summary"].append(this_time - last_time)
                last_time = this_time
        
        if self.cfg.rank == 0:
            for k, v in all_time.items():
                self.echo(f"{k} {np.mean(v):.2f}")
            self.echo(f"# step {self.total_step} # sample {self.total_inst}")
        
    def infer(self, dataloader, samples=float("inf")):
        nsample = 0 
        for batch in self.dataloader:
            images, audios = self.make_batch(batch)
            if nsample > samples:
                continue # iterate through every batch 
            image_features = self.image_head(images)
            audio_features = self.audio_head(audios)
            self.loss_head(image_features, audio_features)    
            nsample += images.shape[0]
        return self.loss_head.report() 

    def save(self):
        fsave = f"{self.cfg.model_root}/{self.cfg.model_name}/{self.total_step:08d}.pth"
        self.echo(f"Saving the checkpoint to {fsave}")
        model = self.model.module if isinstance(self.model, DistributedDataParallel) else self.model
        checkpoint = {
            "cfg": self.cfg, "model": model.collect_state_dict(),
        }
        torch.save(checkpoint, fsave)

    def build_optimizer(self, tunable_params={}):
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

