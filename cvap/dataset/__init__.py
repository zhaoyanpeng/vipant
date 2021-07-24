import abc

from .tfrecord import PairImageSpectrogramTFRecords
from .image_audio import (
    build_dataloader, ImageAudioCollator, ImageAudioDataset, ImageAudioDatasetSrc, ImageAudioDatasetNpz
)

class AbstractTransform(abc.ABC):

    @abc.abstractmethod
    def __call__(self, x):
        pass

    def __repr__(self):
        return self.__class__.__name__ + '()'

