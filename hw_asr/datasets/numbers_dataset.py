import logging
import shutil
import gdown
import pandas as pd

from hw_asr.base.base_dataset import BaseDataset, add_element_to_index
from hw_asr.utils import ROOT_PATH
from hw_asr.utils.parse_config import ConfigParser

logger = logging.getLogger(__name__)

LINK = "https://drive.google.com/uc?id=1HKtLLbiEk0c3l1mKz9LUXRAmKd3DvD0P"


class NumbersDataset(BaseDataset):
    def __init__(self, part, data_dir=None, *args,
                 **kwargs):
        assert part in ("train", "test-example"), "Wrong part for numbers " \
                                                  "dataset"
        if data_dir is None:
            data_dir = ROOT_PATH / "data" / "datasets" / "numbers"
            data_dir.mkdir(exist_ok=True, parents=True)
        self._data_dir = data_dir
        self.all_text_txt_file_path = str(self._data_dir / part /
                                          "all_txt_file.txt")
        index = self._get_or_load_index("train")
        index_test = self._get_or_load_index("test-example")
        if part == "test-example":
            index = index_test
        super().__init__(index, *args, **kwargs)

    def _load(self):
        arch_path = self._data_dir / "numbers.zip"
        print(f"Loading numbers dataset")
        gdown.download(LINK, str(arch_path), quiet=False)
        shutil.unpack_archive(arch_path, self._data_dir)
        for fpath in (self._data_dir / "numbers").iterdir():
            shutil.move(str(fpath), str(self._data_dir / fpath.name))
        shutil.rmtree(str(self._data_dir / "numbers"))

    def _get_or_load_index(self, part):
        index_path = self._data_dir / part / "index.json"
        return self.get_index(index_path, part)

    def _create_index(self, subfolder):
        index = []
        split_dir = self._data_dir / subfolder
        if not split_dir.exists():
            self._load()
        csv_res = pd.read_csv(str(split_dir) + ".csv")
        dict_res = csv_res.set_index("path").to_dict("index")
        all_txt_file = open(str(split_dir / "all_txt_file.txt"), "w")
        for key, value in dict_res.items():
            wav_path = self._data_dir / key
            text = str(value["number"])
            add_element_to_index(all_txt_file, index, text, wav_path)
        all_txt_file.close()
        return index


if __name__ == "__main__":
    config_parser = ConfigParser.get_default_configs()

    ds = NumbersDataset(
        "train", config_parser=config_parser
    )
    item = ds[0]
    print(item)
