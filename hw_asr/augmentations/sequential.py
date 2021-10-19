from typing import List, Callable

from torch import Tensor

from hw_asr.augmentations.base import AugmentationBase


class SequentialAugmentation(AugmentationBase):
    def __init__(self, augmentation_list: List[Callable]):
        self.augmentation_list = augmentation_list

    def __call__(self, data: Tensor, sample_rate):
        x = data
        for augmentation in self.augmentation_list:
            x, sample_rate = augmentation(x, sample_rate)
        return x, sample_rate
