import logging
import os
import shutil
from pathlib import Path

from speechbrain.utils.data_utils import download_file
from tqdm import tqdm

from hw_asr.base.base_dataset import BaseDataset
from hw_asr.utils import ROOT_PATH
from hw_asr.utils.parse_config import ConfigParser
from hw_asr.datasets.utils import add_element_to_index

logger = logging.getLogger(__name__)

URL_LINKS = {
    "dev-clean": "https://www.openslr.org/resources/12/dev-clean.tar.gz",
    "dev-other": "https://www.openslr.org/resources/12/dev-other.tar.gz",
    "test-clean": "https://www.openslr.org/resources/12/test-clean.tar.gz",
    "test-other": "https://www.openslr.org/resources/12/test-other.tar.gz",
    "train-clean-100": "https://www.openslr.org/resources/12/train-clean-100"
                       ".tar.gz",
    "train-clean-360": "https://www.openslr.org/resources/12/train-clean-360"
                       ".tar.gz",
    "train-other-500": "https://www.openslr.org/resources/12/train-other-500"
                       ".tar.gz",
}


class LibrispeechDataset(BaseDataset):
    def __init__(self, part, data_dir=None, create_bpe=False, *args,
                 **kwargs):
        assert part in URL_LINKS or part == 'train_all'

        if data_dir is None:
            data_dir = ROOT_PATH / "data" / "datasets" / "librispeech"
            data_dir.mkdir(exist_ok=True, parents=True)
        else:
            data_dir = Path(data_dir)
        self._data_dir = data_dir
        self.all_text_txt_file_path = None
        self.create_bpe = create_bpe
        if part == 'train_all':
            index = sum([self._get_or_load_index(part)
                         for part in URL_LINKS if 'train' in part], [])
        else:
            index = self._get_or_load_index(part)

        super().__init__(index, *args, **kwargs)
        self.all_text_txt_file_path = str(self._data_dir / part /
                                          "all_txt_file.txt")

    def _load_part(self, part):
        arch_path = self._data_dir / f"{part}.tar.gz"
        print(f"Loading part {part}")
        download_file(URL_LINKS[part], arch_path)
        shutil.unpack_archive(arch_path, self._data_dir)
        for fpath in (self._data_dir / "LibriSpeech").iterdir():
            shutil.move(str(fpath), str(self._data_dir / fpath.name))
        os.remove(str(arch_path))
        shutil.rmtree(str(self._data_dir / "LibriSpeech"))

    def _get_or_load_index(self, part):
        index_path = self._data_dir / f"{part}_index.json"
        return self.get_index(index_path, part)

    def _create_index(self, part):
        index = []
        split_dir = self._data_dir / part
        if not split_dir.exists():
            self._load_part(part)

        flac_dirs = set()
        for dirpath, dirnames, filenames in os.walk(str(split_dir)):
            if any([f.endswith(".flac") for f in filenames]):
                flac_dirs.add(dirpath)
        if self.create_bpe:
            all_txt_file = open(str(split_dir / "all_txt_file.txt"), "w")
        else:
            all_txt_file = None
        for flac_dir in tqdm(
                list(flac_dirs), desc=f"Preparing librispeech folders: {part}"
        ):
            flac_dir = Path(flac_dir)
            trans_path = list(flac_dir.glob("*.trans.txt"))[0]
            with trans_path.open() as f:
                for line in f:
                    f_id = line.split()[0]
                    f_text = " ".join(line.split()[1:]).strip()
                    flac_path = flac_dir / f"{f_id}.flac"
                    add_element_to_index(all_txt_file, index, f_text,
                                         flac_path)
        if all_txt_file is not None:
            all_txt_file.close()
        return index


if __name__ == "__main__":
    config_parser = ConfigParser.get_default_configs()

    ds = LibrispeechDataset(
        "dev-clean", config_parser=config_parser
    )
    item = ds[0]
    print(item)
