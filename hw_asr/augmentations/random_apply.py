import random
from typing import Callable

from torch import Tensor


class RandomApply:
    def __init__(self, augmentation: Callable, p: float):
        assert 0 <= p <= 1
        self.augmentation = augmentation
        self.p = p

    def __call__(self, data: Tensor, sample_rate):
        if random.random() < self.p:
            return self.augmentation(data, sample_rate)
        else:
            return data, sample_rate
