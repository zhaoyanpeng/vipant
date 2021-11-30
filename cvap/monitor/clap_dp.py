from omegaconf import OmegaConf
import os, re
from collections import defaultdict

import time
import torch
import numpy as np
from torch import nn

import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import data_parallel
from torch.nn.parallel import DistributedDataParallel

from ..util import numel
from ..model import build_main_model, extract_model_file
from ..module import LARS, exclude_bias_or_norm, adjust_learning_rate

from .cvap_dp import Monitor

class Monitor(Monitor):
    def __init__(self, cfg, echo, device):
        super(Monitor, self).__init__(cfg, echo, device)

    def eval_norms(self):
        self.echo("Evaluate mean and std...")
        cnt = 0.
        som = torch.tensor(0., device=self.device)
        sos = torch.tensor(0., device=self.device)
        som_list, sos_list = list(), list()
        for step, batch in enumerate(self.dataloader):
            audios, _, _ = self.make_batch(batch)
            bsz = audios.shape[0]
            new_cnt = cnt + bsz
            mean = audios.mean(axis=[2, 3])
            mean_sq = (audios ** 2).mean(axis=[2, 3])
            """ incremental update might be numerically unstable
            som = (som * cnt + mean.sum()) / new_cnt
            sos = (sos * cnt + mean_sq.sum()) / new_cnt
            """
            som_list.append(mean)
            sos_list.append(mean_sq)
        som = torch.cat(som_list, 0).mean(axis=[0])
        sos = torch.cat(sos_list, 0).mean(axis=[0])
        std = (sos - som ** 2).sqrt()
        self.echo(f"MEAN: {som.cpu().tolist()} STD: {std.cpu().tolist()}")

    def encode_text(self):
        """ use batch size 1 in case each audio clip has different numbers of captions.
        """
        rcfg = self.cfg.running
        model_file = rcfg.clip_model_name.lower() #cfg.model_file
        audio_root = f"{rcfg.data_root}/caption/audiocap/{model_file}"
        if not os.path.exists(audio_root):
            os.makedirs(audio_root)
        def save_npz(names, text):
            assert text.shape[0] % len(names) == 0, "please use batch size 1."
            ncap_per_audio = int(text.shape[0] / len(names))
            for i, name in enumerate(names):
                np.savez_compressed(
                    f"{audio_root}/{name}", v=text[i * ncap_per_audio : (i + 1) * ncap_per_audio]
                )
        self.echo(f"Encode caption ({len(self.dataloader)} batches)...to `{audio_root}`")
        nsample = 0
        start_time = time.time()
        device_ids = [i for i in range(self.cfg.num_gpus)]
        for step, batch in enumerate(self.dataloader):
            _, text, names = self.make_batch(batch)

            text_features = self.model.encode_text(text, device_ids=device_ids)
            text_features = text_features.cpu().numpy()

            save_npz(names, text=text_features)

            nsample += text.shape[0]
            if (step + 1) % rcfg.peep_rate == 0:
                self.echo(f"--step {step + 1:08d} {nsample / (time.time() - start_time):.2f} samples/s")
        self.echo(f"Saving {nsample} text vectors.")

    def build_data(self):
        rcfg = self.cfg.running
        if rcfg.dataloader == "al":
            from ..dataset import build_audio_text_dataloader as build_dataloader
        elif rcfg.dataloader == "lv":
            from ..dataset import build_image_text_dataloader as build_dataloader
        else:
            raise ValueError("Unknown data loader `{rcfg.dataloader}`.")
        data_name = rcfg.eval_name if self.cfg.eval else rcfg.data_name
        _, self.dataloader = build_dataloader(
            self.cfg, data_name, shuffle=(not self.cfg.eval), train=(not self.cfg.eval)
        )
        nstep = len(self.dataloader)
        if nstep < self.cfg.running.peep_rate:
            self.cfg.running.peep_rate = nstep
        self.echo(f"Instantiate main dataloader from `{data_name}': total {nstep} ({self.cfg.running.peep_rate}) batches.")
        self.gold_file = f"{rcfg.data_root}/{data_name}.csv"
        # evaluation
        eval_name = "IGNORE_ME" if self.cfg.eval else rcfg.eval_name
        data_path = f"{rcfg.data_root}/{eval_name}"
        do_eval = os.path.isdir(data_path) or os.path.isfile(f"{data_path}.csv") #or tf.io.gfile.exists(f"{data_path}.csv")
        _, self.evalloader = build_dataloader(
            self.cfg, eval_name, shuffle=False, train=False
        ) if do_eval else (None, None)
        if self.evalloader is not None:
            self.echo(f"Will do evaluation every {rcfg.save_rate} steps on {len(self.evalloader)} batches ({eval_name}).")
            self.gold_file = f"{rcfg.data_root}/{eval_name}.csv"
        # test
        test_name = "IGNORE_ME" if self.cfg.eval else rcfg.test_name
        data_path = f"{rcfg.data_root}/{test_name}"
        do_eval = os.path.isdir(data_path) or os.path.isfile(f"{data_path}.csv") #or tf.io.gfile.exists(f"{data_path}.csv")
        _, self.testloader = build_dataloader(
            self.cfg, test_name, shuffle=False, train=False,
        ) if do_eval else (None, None)
        if self.testloader is not None:
            self.echo(f"Will do test every {rcfg.save_rate} steps on {len(self.testloader)} batches ({test_name}).")
            self.gold_file_test = f"{rcfg.data_root}/{test_name}.csv"

    def learn(self):
        if self.cfg.running.audio.eval_norms:
            return # `eval_norms` is the only task
        if not self.model.training:
            self.echo("Evaluating started...")

            #with torch.no_grad():
            #    self.encode_text()
            #return None

            if self.cfg.model_file.endswith(".out"):
                with torch.no_grad():
                    self.repeated_retrieval() # multiple evaluations
            else:
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
        if len(batch) > 3: # needed when using `AudioCapDatasetSrc`
            return ( # (image, audio, text, label, name)
                torch.tensor(
                    batch[1], device=self.device
                ).unsqueeze(1), # audio
                torch.tensor(
                    batch[2], device=self.device
                ), # captions / label
                batch[4], # captions / label name
            )
        elif len(batch) == 3:
            batch = (
                torch.tensor(
                    batch[0], device=self.device
                ).unsqueeze(1) if len(batch[0].shape) == 3 else (
                    torch.tensor(batch[0], device=self.device)
                ), # audio
                torch.tensor(
                    batch[1], device=self.device
                ), # captions / label
                batch[2], # captions / label name
            )
        else:
            raise ValueError(f"I do not know how to parse `batch` (w/ {len(batch)} items).")
        return batch # bare tensors

    def epoch(self, iepoch):
        all_time = defaultdict(list)
        self.timeit(all_time)        
        device_ids = [i for i in range(self.cfg.num_gpus)]
        nchunk = dist.get_world_size() if torch.distributed.is_initialized() else 1  
        warmup_step_rate = max(self.cfg.optimizer.warmup_steps // 20, 1)
        for step, batch in enumerate(self.dataloader, start=iepoch * len(self.dataloader)):
            audios, text, _ = self.make_batch(batch)
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
                loss = self.model(audios, text, device_ids=device_ids, retrieval=self.cfg.running.retrieval)
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
            self.total_inst += audios.shape[0] * nchunk
            if force_eval or (self.cfg.rank == 0 and self.total_step % self.cfg.running.peep_rate == 0):
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
            if force_eval or self.total_step % self.cfg.running.save_rate == 0 or (
                    self.cfg.running.save_epoch and self.total_step % len(self.dataloader) == 0
                ): # distributed eval
                report = ""
                if self.evalloader is not None and loss.detach() < 5.: # no need to eval if CE is too large
                    self.model.train(False)
                    with torch.no_grad():
                        report = self.infer(
                            self.evalloader, samples=self.cfg.running.eval_samples, iepoch=iepoch
                        )
                    self.model.train(True)
                if report != "":
                    self.echo(f"{report}")

                report = ""
                if self.testloader is not None and loss.detach() < 5.: # no need to eval if CE is too large
                    self.model.train(False)
                    with torch.no_grad():
                        report = self.infer(
                            self.testloader, samples=self.cfg.running.test_samples, iepoch=iepoch
                        )
                    self.model.train(True)
                if report != "":
                    self.echo(f"{report}")

                if self.cfg.rank == 0:
                    self.save()
            self.timeit(all_time, key="report")

        if not self.cfg.optimizer.use_lars:
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
            audios, text, names = self.make_batch(batch)
            #msg = f"{audios[0, 0, 50, 50:55]} {text[0, 50, 50:55]}" # if ibatch == 0 else ""
            #print(f"{nsample}\t{ibatch}/{nbatch} done {msg}")
            loss = self.model(audios, text, device_ids=device_ids, names=names, retrieval=self.cfg.running.retrieval)
            nsample += audios.shape[0] * nchunk
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

    def repeated_retrieval(self):
        self.echo("Evaluate multiple checkpoints.")
        model_files = extract_model_file(self.cfg, self.echo)
        for model_file in model_files:
            self.cfg.model_file = model_file # modify the global
            tunable_params = self.model.build()
            self.model.train(not self.cfg.eval)

            report = self.infer(self.dataloader, samples=self.cfg.running.eval_samples)
            self.echo(f"{report}")

