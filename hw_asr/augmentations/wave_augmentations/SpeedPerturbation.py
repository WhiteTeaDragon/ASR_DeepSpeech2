from torch import Tensor

from hw_asr.augmentations.base import AugmentationBase

import torchaudio
import random


class SpeedPerturbation(AugmentationBase):
    def __init__(self, speed_min, speed_max, *args, **kwargs):
        self.speed_min = speed_min
        self.speed_max = speed_max

    def __call__(self, data: Tensor, sample_rate):
        if random.random() > 0.5:
            new_sample_rate = int(self.speed_max * sample_rate)
        else:
            new_sample_rate = int(self.speed_min * sample_rate)
        x = torchaudio.functional.resample(data, sample_rate, new_sample_rate)
        return x, new_sample_rate
