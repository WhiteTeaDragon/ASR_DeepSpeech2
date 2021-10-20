import logging
import librosa

from datasets import load_dataset

from hw_asr.base.base_dataset import BaseDataset
from hw_asr.utils import ROOT_PATH
from hw_asr.utils.parse_config import ConfigParser
from hw_asr.base.base_text_encoder import BaseTextEncoder

logger = logging.getLogger(__name__)


def _load():
    return load_dataset("lj_speech")


class LJDataset(BaseDataset):
    def __init__(self, data_dir=None, *args, **kwargs):
        if data_dir is None:
            data_dir = ROOT_PATH / "data" / "datasets" / "lj"
            data_dir.mkdir(exist_ok=True, parents=True)
        self._data_dir = data_dir
        self.all_text_txt_file_path = None
        index = self._get_or_load_index()
        super().__init__(index, *args, **kwargs)

    def _get_or_load_index(self):
        index = _load()["train"]
        new_index = []
        self.all_text_txt_file_path = str(self._data_dir / "all_txt_file.txt")
        all_txt_file = open(self.all_text_txt_file_path, "w")
        for entry in index:
            new_entry = {"audio_len": librosa.get_duration(
                filename=entry["file"]),
                         "path": entry["file"],
                         "text": BaseTextEncoder.normalize_text(
                             entry["normalized_text"].lower())}
            new_index.append(new_entry)
            print(new_entry["text"], file=all_txt_file)
        all_txt_file.close()
        return new_index


if __name__ == "__main__":
    config_parser = ConfigParser.get_default_configs()

    ds = LJDataset(
        "dev-clean", config_parser=config_parser
    )
    item = ds[0]
    print(item)
