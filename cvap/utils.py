import os
import math
import random
import logging
import numpy, torch
import torch.distributed as dist

def seed_all_rng(seed):
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)

def setup_logger(output_dir=None, name="cvap", rank=0, output=None):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    if rank == 0:
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        console.setFormatter(formatter)
        logger.addHandler(console)
    if os.path.exists(output_dir):
        logger.info(f'Warning: the folder {output_dir} exists.')
    else:
        logger.info(f'Creating {output_dir}')
        if rank == 0: 
            os.makedirs(output_dir)
    if torch.distributed.is_initialized():
        dist.barrier() # output dir should have been ready
    if output is not None:
        filename = os.path.join(output_dir, f'train_{rank}.out')
        handler = logging.FileHandler(filename, 'w')
        handler.setLevel(logging.INFO)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

def exclude_bias_or_norm(p):
    return p.ndim < 2 

def adjust_learning_rate(cfg, optimizer, dataloader, step):
    max_steps = cfg.epochs * len(dataloader)
    warmup_steps = 10 * len(dataloader)
    base_lr = cfg.batch_size / 256
    if step < warmup_steps:
        lr = base_lr * step / warmup_steps
    else:
        step -= warmup_steps
        max_steps -= warmup_steps
        q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
        end_lr = base_lr * 0.001
        lr = base_lr * q + end_lr * (1 - q)
    optimizer.param_groups[0]['lr'] = lr * cfg.lr_weight
    optimizer.param_groups[1]['lr'] = lr * cfg.lr_bias

