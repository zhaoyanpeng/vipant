from .loss_head import build_loss_head, LOSS_HEADS_REGISTRY
from .loss_more import (
    BCELossHead, BCEAndCELossHead, ImaginedCLFLossHead
)
LOSS_HEADS_REGISTRY.register(BCELossHead)
LOSS_HEADS_REGISTRY.register(BCEAndCELossHead)
LOSS_HEADS_REGISTRY.register(ImaginedCLFLossHead)
