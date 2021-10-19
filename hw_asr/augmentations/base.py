from torch import Tensor


class AugmentationBase:
    def __call__(self, data: Tensor, sample_rate) -> Tensor:
        raise NotImplementedError
