import abc

class AbstractTransform(abc.ABC):

    @abc.abstractmethod
    def __call__(self, x):
        pass

    def __repr__(self):
        return self.__class__.__name__ + '()'

from .tfrecord import PairImageSpectrogramTFRecords
from .image_audio import (
    build_dataloader, ImageAudioCollator, ImageAudioDataset, ImageAudioDatasetSrc, ImageAudioDatasetNpz
)
from .audio_text import (
    build_audio_text_dataloader
)
from .audioset import (
    build_audioset_dataloader, build_audioset_label_map
)
