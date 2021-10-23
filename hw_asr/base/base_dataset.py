import logging
import random

import numpy as np
import torch
import torchaudio
import json
from torch import Tensor
from torch.utils.data import Dataset

from hw_asr.base.base_text_encoder import BaseTextEncoder
from hw_asr.utils.parse_config import ConfigParser
from hw_asr.text_encoder.ctc_char_text_encoder import CTCCharTextEncoder

logger = logging.getLogger(__name__)


class BaseDataset(Dataset):
    def __init__(
        self,
        index,
        config_parser: ConfigParser,
        wave_augs=None,
        spec_augs=None,
        limit=None,
        max_audio_length=None,
        max_text_length=None,
        min_audio_length=None,
        min_text_length=None
    ):
        self.text_encoder = CTCCharTextEncoder.get_simple_alphabet()
        self.config_parser = config_parser
        self.wave_augs = wave_augs
        self.spec_augs = spec_augs

        for entry in index:
            assert "audio_len" in entry, (
                "Each dataset item should include field 'audio_len'"
                " - duration of audio (in seconds)."
            )
            assert "path" in entry, (
                "Each dataset item should include field 'path'" "- path to "
                "audio file. "
            )
            assert "text" in entry, (
                "Each dataset item should include field 'text'"
                " - text transcription of the audio."
            )

        index = self._filter_records_from_dataset(
            index, max_audio_length, max_text_length, min_audio_length,
            min_text_length, limit
        )

        # it's a good idea to sort index by audio length
        # It would be easier to write length-based batch samplers later
        index = self._sort_index(index)
        self._index = index

    def __getitem__(self, ind):
        data_dict = self._index[ind]
        audio_path = data_dict["path"]
        audio_wave = self.load_audio(audio_path)
        audio_wave, audio_spec, sample_rate = self.process_wave(audio_wave)
        return {
            "audio": audio_wave,
            "spectrogram": audio_spec,
            "duration": data_dict["audio_len"],
            "text": data_dict["text"].replace("'", ""),
            "text_encoded": self.text_encoder.encode(data_dict["text"].replace(
                "'", "")),
            "audio_path": audio_path,
            "sample_rate": sample_rate
        }

    @staticmethod
    def _sort_index(index):
        return sorted(index, key=lambda x: x["audio_len"])

    def __len__(self):
        return len(self._index)

    def load_audio(self, path):
        audio_tensor, sr = torchaudio.load(path)
        # remove all channels but the first
        audio_tensor = audio_tensor[0:1, :]
        target_sr = self.config_parser["preprocessing"]["sr"]
        if sr != target_sr:
            audio_tensor = torchaudio.functional.resample(audio_tensor, sr,
                                                          target_sr)
        return audio_tensor

    def process_wave(self, audio_tensor_wave: Tensor):
        sr = self.config_parser["preprocessing"]["sr"]
        with torch.no_grad():
            if self.wave_augs is not None:
                audio_tensor_wave, sr = self.wave_augs(audio_tensor_wave, sr)
            wave2spec = self.config_parser.init_obj(
                self.config_parser["preprocessing"]["spectrogram"],
                torchaudio.transforms,
            )
            audio_tensor_spec = wave2spec(audio_tensor_wave)
            if self.spec_augs is not None:
                audio_tensor_spec, sr = self.spec_augs(
                    audio_tensor_spec, sr)
            return audio_tensor_wave, audio_tensor_spec, sr

    @staticmethod
    def _filter_records_from_dataset(
        index: list, max_audio_length, max_text_length, min_audio_length,
            min_text_length, limit
    ) -> list:
        initial_size = len(index)
        if max_audio_length is not None:
            exceeds_audio_length = np.array(
                [el["audio_len"] for el in index]) >= max_audio_length
            _total = exceeds_audio_length.sum()
            logger.info(
                f"{_total} ({_total / initial_size:.1%}) records are longer "
                f"then {max_audio_length} seconds. Excluding them."
            )
        else:
            exceeds_audio_length = False

        initial_size = len(index)
        if min_audio_length is not None:
            short_audio_length = np.array(
                [el["audio_len"] for el in index]) <= min_audio_length
            _total = short_audio_length.sum()
            logger.info(
                f"{_total} ({_total / initial_size:.1%}) records are shorter "
                f"then {min_audio_length} seconds. Excluding them."
            )
        else:
            short_audio_length = False

        initial_size = len(index)
        if max_text_length is not None:
            exceeds_text_length = np.array(
                [
                    len(BaseTextEncoder.normalize_text(el["text"]))
                    for el in index]) >= max_text_length
            _total = exceeds_text_length.sum()
            logger.info(
                f"{_total} ({_total / initial_size:.1%}) records are longer "
                f"then {max_text_length} characters. Excluding them."
            )
        else:
            exceeds_text_length = False

        initial_size = len(index)
        if min_text_length is not None:
            short_text_length = np.array(
                [
                    len(BaseTextEncoder.normalize_text(el["text"]))
                    for el in index]) <= min_text_length
            _total = exceeds_text_length.sum()
            logger.info(
                f"{_total} ({_total / initial_size:.1%}) records are shorter "
                f"then {min_text_length} characters. Excluding them."
            )
        else:
            short_text_length = False

        records_to_filter = exceeds_text_length | exceeds_audio_length\
                            | short_text_length | short_audio_length

        if records_to_filter is not False and records_to_filter.any():
            _total = records_to_filter.sum()
            index = [el for el, exclude in zip(index, records_to_filter) if
                     not exclude]
            logger.info(
                f"Filtered {_total}({_total / initial_size:.1%}) records "
                f"from dataset "
            )

        if limit is not None:
            random.seed(42)  # best seed for deep learning
            random.shuffle(index)
            index = index[:limit]
        return index

    def set_text_encoder(self, text_encoder):
        self.text_encoder = text_encoder

    def get_index(self, index_path, subfolder):
        if index_path.exists():
            with index_path.open() as f:
                index = json.load(f)
        else:
            index = self._create_index(subfolder)
            with index_path.open("w") as f:
                json.dump(index, f, indent=2)
        return index

    def _create_index(self, subfolder):
        pass
