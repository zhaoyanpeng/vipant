from .helper import *
from .audioset_classifier import AudioSetClassifier
from .audio_classifier import AudioClassifier
from .ast import ASTClassifier
from .cvap_ddp import CVAPDDP
from .cvalp_dp import CVALPDP
from .clap_dp import CLAPDP
from .cvap_dp import CVAPDP
from .cvap_siamese import CVASPDP

from fvcore.common.registry import Registry

VAL_MODELS_REGISTRY = Registry("VAL_MODELS")
VAL_MODELS_REGISTRY.__doc__ = """
Registry for vision-audio-language models.
"""

VAL_MODELS_REGISTRY.register(AudioSetClassifier)
VAL_MODELS_REGISTRY.register(AudioClassifier)
VAL_MODELS_REGISTRY.register(ASTClassifier)
VAL_MODELS_REGISTRY.register(CVAPDDP)
VAL_MODELS_REGISTRY.register(CVALPDP)
VAL_MODELS_REGISTRY.register(CVASPDP)
VAL_MODELS_REGISTRY.register(CLAPDP)
VAL_MODELS_REGISTRY.register(CVAPDP)

def build_main_model(cfg, echo, **kwargs):
    return VAL_MODELS_REGISTRY.get(cfg.worker)(cfg, echo)
