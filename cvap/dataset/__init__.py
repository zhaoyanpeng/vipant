import abc

class AbstractTransform(abc.ABC):

    @abc.abstractmethod
    def __call__(self, x):
        pass

    def __repr__(self):
        return self.__class__.__name__ + '()'

from .image_audio import (
    build_dataloader, ImageAudioCollator, ImageAudioDatasetSrc, ImageAudioDatasetNpz
)
from .audio_text import (
    build_audio_text_dataloader
)
from .image_text import (
    build_image_text_dataloader
)
from .audioset import (
    build_audioset_dataloader, build_audioset_label_map, build_filter_set
)
from .audioset_ast import (
    build_ast_dataloader
)
