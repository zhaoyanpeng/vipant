import os
import logging
import random
import numpy
import torch
import torch.distributed as dist
import jax.numpy as jnp

from .function import make_dummy_image_with_text

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

def numel(model: torch.nn.Module, trainable: bool = False):
    parameters = list(model.parameters())
    if trainable:
        parameters = [p for p in parameters if p.requires_grad]
    unique = {p.data_ptr(): p for p in parameters}.values()
    return sum(p.numel() for p in unique) 

def detect_nan(x):
    return torch.isnan(x).any(), torch.isinf(x).any()

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = self.count = 0

    def __call__(self, val, n=1):
        self.count += n
        self.sum += val * n

    @property
    def average(self):
        return self.sum / self.count

def unit_normalize(x):
    """
    Normalize `x` to have unit norm over the final dimension
    :param x:
    :return:
    """
    x_f32 = x.astype(jnp.float32)
    x_norm = x_f32 / jnp.sqrt(jnp.square(x_f32).sum(-1, keepdims=True) + 1e-5)
    return x_norm.astype(x.dtype)
