import torch_audiomentations
import shutil

from torch import Tensor
from speechbrain.utils.data_utils import download_file

from hw_asr.augmentations.base import AugmentationBase
from hw_asr.utils import ROOT_PATH

# https://zenodo.org/record/2529934/files/FSDnoisy18k.audio_test.zip


class Noise(AugmentationBase):
    def __init__(self, noise_url):
        data_dir = ROOT_PATH / "data" / "noise"
        data_dir.mkdir(exist_ok=True, parents=True)
        arch_path = data_dir / "FSDnoisy18k.audio_test.zip"
        print(f"Loading noise")
        download_file(noise_url, arch_path)
        shutil.unpack_archive(arch_path, data_dir)
        for fpath in (data_dir / "FSDnoisy18k.audio_test").iterdir():
            shutil.move(str(fpath), str(data_dir / fpath.name))
        shutil.rmtree(str(data_dir / "FSDnoisy18k.audio_test"))
        self._aug = torch_audiomentations.AddBackgroundNoise(str(data_dir))

    def __call__(self, data: Tensor, sample_rate):
        x = data.unsqueeze(1)
        return self._aug(x, sample_rate).squeeze(1), sample_rate
