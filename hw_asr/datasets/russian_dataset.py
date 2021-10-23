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
    "public_youtube1120_hq": "https://zenodo.org/record/4899208/files"
                             "/public_youtube1120_hq.tar.gz",
    "public_youtube700_val": "https://zenodo.org/record/4899885/files"
                             "/public_youtube700_val.tar.gz"
}


class RussianDataset(BaseDataset):
    def __init__(self, part, data_dir=None, create_bpe=False, *args,
                 **kwargs):
        assert part in URL_LINKS

        if data_dir is None:
            data_dir = ROOT_PATH / "data" / "datasets" / "russian"
            data_dir.mkdir(exist_ok=True, parents=True)
        else:
            data_dir = Path(data_dir)
        self._data_dir = data_dir
        self.all_text_txt_file_path = None
        self.create_bpe = create_bpe
        index = self._get_or_load_index(part)

        super().__init__(index, *args, **kwargs)
        self.all_text_txt_file_path = str(self._data_dir / part /
                                          "all_txt_file.txt")

    def _load_part(self, part):
        arch_path = self._data_dir / f"{part}.tar.gz"
        print(f"Loading part {part}")
        download_file(URL_LINKS[part], arch_path)
        shutil.unpack_archive(arch_path, self._data_dir)
        os.remove(str(arch_path))

    def _get_or_load_index(self, part):
        index_path = self._data_dir / f"{part}_index.json"
        return self.get_index(index_path, part)

    def _create_index(self, part):
        index = []
        split_dir = self._data_dir / part
        if not split_dir.exists():
            self._load_part(part)

        audio_dirs = set()
        for dirpath, dirnames, filenames in os.walk(str(split_dir)):
            for f in filenames:
                if f.endswith(".opus"):
                    file_ext = "opus"
                    audio_dirs.add(dirpath)
                    break
                if f.endswith(".wav"):
                    file_ext = "wav"
                    audio_dirs.add(dirpath)
                    break
        if len(audio_dirs) == 0:
            raise ValueError("No audios in the dataset")
        if self.create_bpe:
            all_txt_file = open(str(split_dir / "all_txt_file.txt"), "w")
        else:
            all_txt_file = None
        for flac_dir in tqdm(
                list(audio_dirs), desc=f"Preparing russian dataset"
                                       f" folders: {part}"
        ):
            flac_dir = Path(flac_dir)
            trans_path = list(flac_dir.glob("*.txt"))
            for i in range(len(trans_path)):
                curr_path = trans_path[i]
                with curr_path.open() as f:
                    text = " ".join(f.readlines())
                audio_path = curr_path.with_suffix(file_ext)
                f_text = " ".join(text.split()).strip()
                add_element_to_index(all_txt_file, index, f_text,
                                     audio_path)
        if all_txt_file is not None:
            all_txt_file.close()
        return index


if __name__ == "__main__":
    config_parser = ConfigParser.get_default_configs()

    ds = RussianDataset(
        "public_youtube700_val", config_parser=config_parser
    )
    item = ds[0]
    print(item)
