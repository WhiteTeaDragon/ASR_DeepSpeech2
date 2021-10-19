import torchvision
from torch import Tensor

from hw_asr.augmentations.base import AugmentationBase


class RandomErasing(AugmentationBase):
    def __init__(self, *args, **kwargs):
        self._aug = torchvision.transforms.RandomErasing(*args, **kwargs)

    def __call__(self, data: Tensor, sample_rate):
        return self._aug(data)
