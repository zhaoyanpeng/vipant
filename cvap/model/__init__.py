from .helper import *
from .audioset_clf import ASClassifier
from .esc50_clf import ESClassifier
from .cvalp import CVALP
from .clap import CLAP
from .clvp import CLVP
from .cvap import CVAP
from .siamese_va import CVASP

from fvcore.common.registry import Registry

VAL_MODELS_REGISTRY = Registry("VAL_MODELS")
VAL_MODELS_REGISTRY.__doc__ = """
Registry for vision-audio-language models.
"""

VAL_MODELS_REGISTRY.register(ASClassifier)
VAL_MODELS_REGISTRY.register(ESClassifier)
VAL_MODELS_REGISTRY.register(CVALP)
VAL_MODELS_REGISTRY.register(CVASP)
VAL_MODELS_REGISTRY.register(CLAP)
VAL_MODELS_REGISTRY.register(CLVP)
VAL_MODELS_REGISTRY.register(CVAP)

def build_main_model(cfg, echo, **kwargs):
    return VAL_MODELS_REGISTRY.get(cfg.worker)(cfg, echo)
