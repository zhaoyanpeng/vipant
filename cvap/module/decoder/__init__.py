from .loss_head import build_loss_head, LOSS_HEADS_REGISTRY
from .loss_more import (
    LMLossHead, BCELossHead, BCHingeLossHead, ImagineAndClassifyLossHead
)
LOSS_HEADS_REGISTRY.register(LMLossHead)
LOSS_HEADS_REGISTRY.register(BCELossHead)
LOSS_HEADS_REGISTRY.register(BCHingeLossHead)
LOSS_HEADS_REGISTRY.register(ImagineAndClassifyLossHead)
