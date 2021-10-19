from torch import Tensor

from hw_asr.augmentations.base import AugmentationBase

import torchaudio


class SpeedPerturbation(AugmentationBase):
    def __init__(self, speed_x, *args, **kwargs):
        self.speed_x = speed_x

    def __call__(self, data: Tensor, sample_rate):
        x = torchaudio.functional.resample(data, sample_rate,
                                           self.speed_x * sample_rate)
        return x
