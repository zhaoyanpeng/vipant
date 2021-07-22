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

