import torch_audiomentations
from torch import Tensor

from hw_asr.augmentations.base import AugmentationBase


class Gain(AugmentationBase):
    def __init__(self, *args, **kwargs):
        self._aug = torch_audiomentations.Gain(*args, **kwargs)

    def __call__(self, data: Tensor, sample_rate):
        x = data.unsqueeze(1)
        return self._aug(x).squeeze(1), sample_rate
