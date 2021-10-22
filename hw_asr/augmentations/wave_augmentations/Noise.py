import shutil
import random
import torchaudio
import torch
import math

from torch import Tensor
from speechbrain.utils.data_utils import download_file

from hw_asr.augmentations.base import AugmentationBase
from hw_asr.utils import ROOT_PATH

# https://zenodo.org/record/2529934/files/FSDnoisy18k.audio_test.zip


class Noise(AugmentationBase):
    def __init__(self, noise_url, noise_level=20):
        data_dir = ROOT_PATH / "data" / "noise"
        data_dir.mkdir(exist_ok=True, parents=True)
        arch_path = data_dir / "FSDnoisy18k.audio_test.zip"
        self.noise_level = noise_level
        print(f"Loading noise")
        if not arch_path.exists():
            download_file(noise_url, arch_path)
            shutil.unpack_archive(arch_path, data_dir)
            self.file_paths = []
            for fpath in (data_dir / "FSDnoisy18k.audio_test").iterdir():
                filename = str(data_dir / fpath.name)
                shutil.move(str(fpath), filename)
                self.file_paths.append(filename)
            shutil.rmtree(str(data_dir / "FSDnoisy18k.audio_test"))

    def __call__(self, data: Tensor, sample_rate):
        _, len_audio = data.shape
        curr_file = random.choice(self.file_paths)
        noise, _ = torchaudio.load(curr_file)
        noise = noise[0:1, :]
        _, len_noise = noise.shape
        noise_energy = torch.norm(noise)
        audio_energy = torch.norm(data)

        alpha = (audio_energy / noise_energy) * math.pow(
            10, -self.noise_level / 20)
        if len_noise > len_audio:
            noise = noise[:, :len_audio]
        else:
            times = (len_audio + len_noise - 1) // len_noise
            noise = noise.repeat(1, times)[:, :len_audio]

        augmented_wav = data + alpha * noise

        augmented_wav = torch.clamp(augmented_wav, -1, 1)
        return augmented_wav, sample_rate
