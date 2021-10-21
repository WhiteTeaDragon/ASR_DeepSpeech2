import torch
import torchaudio
from torch import Tensor

from hw_asr.augmentations.base import AugmentationBase


class Cutout(AugmentationBase):
    def __init__(self):
        self.aug = torch.nn.Sequential(
            torchaudio.transforms.FrequencyMasking(50),
            torchaudio.transforms.TimeMasking(120),
        )

    def __call__(self, data: Tensor, sample_rate):
        return self.aug(data), sample_rate
