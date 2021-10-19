from torch import Tensor

from hw_asr.augmentations.base import AugmentationBase

import torchaudio
import random


class SpeedPerturbation(AugmentationBase):
    def __init__(self, speed_min, speed_max, *args, **kwargs):
        self.speed_min = speed_min
        self.speed_max = speed_max

    def __call__(self, data: Tensor, sample_rate):
        new_speed = self.speed_max
        if random.random() < 0.5:
            new_speed = self.speed_min
        effects = [
            ["speed", str(new_speed)],  # reduce the speed
            ["rate", f"{sample_rate}"],
        ]
        x, sample_rate = torchaudio.sox_effects.apply_effects_tensor(
            data, sample_rate, effects)
        return x, sample_rate
