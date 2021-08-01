from .transformer import Transformer # has to be the first because the following depend on it
# better API
from .val import build_encoder_module, ENCODER_MODULES_REGISTRY, interp_clip_vp_embedding
# deprecated API
from .vit import VisualTransformer
from .txt import TextualTransformer
from .resnet import ModifiedResNet
# optimizer
from .lars import * 
# encoder heads
from .encoder import *
from .decoder import *
