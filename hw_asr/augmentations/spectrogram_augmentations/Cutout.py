import torch
import torchaudio
from torch import Tensor

from hw_asr.augmentations.base import AugmentationBase


class Cutout(AugmentationBase):
    def __init__(self, freq=50, time=120):
        self.aug = torch.nn.Sequential(
            torchaudio.transforms.FrequencyMasking(freq),
            torchaudio.transforms.TimeMasking(time),
        )

    def __call__(self, data: Tensor, sample_rate):
        return self.aug(data), sample_rate
