from .image_head import build_image_head, IMAGE_HEADS_REGISTRY
from .audio_head import build_audio_head, AUDIO_HEADS_REGISTRY
from .text_head import build_text_head, TEXT_HEADS_REGISTRY
# heads initialized from CLIP
from .clip_head import (
    CLIPImageHead, CLIPAudioHead, CLIPTextHead
)
IMAGE_HEADS_REGISTRY.register(CLIPImageHead)
AUDIO_HEADS_REGISTRY.register(CLIPAudioHead)
TEXT_HEADS_REGISTRY.register(CLIPTextHead )
