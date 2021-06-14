from omegaconf import DictConfig, OmegaConf
import torch
import hydra

import jax
from jax import random, numpy as np, value_and_grad, jit, tree_util
from optax import (
    chain,
    clip_by_global_norm,
    scale_by_adam,
    scale,
    apply_updates,
    add_decayed_weights,
    masked,
)

from cvap.cvap import Monitor
from cvap.utils import seed_all_rng, setup_logger


@hydra.main(config_path="configs", config_name="default")
def train(cfg: DictConfig) -> None:
    cfg_str = OmegaConf.to_yaml(cfg)
    seed_all_rng(cfg.seed)

    rank = 0
    ngpu = torch.cuda.device_count()
    output_dir = f"{cfg.model_root}/{cfg.model_name}"
    logger = setup_logger(
        output_dir=output_dir, rank=rank, output=output_dir,
    )
    logger.info(f"\n\n{cfg_str}")
    logger.info("World size: {}; rank: {}".format(ngpu, rank))

    monitor = Monitor(cfg, logger.info)
    monitor.learn()

if __name__ == "__main__":
    train()