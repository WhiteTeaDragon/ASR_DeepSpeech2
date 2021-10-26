import logging
import csv
from pathlib import Path

from hw_asr.datasets.custom_audio_dataset import CustomAudioDataset
from hw_asr.text_encoder.ctc_char_text_encoder import CTCCharTextEncoder
from hw_asr.text_encoder.char_text_encoder import BaseTextEncoder
from hw_asr.utils.parse_config import ConfigParser

logger = logging.getLogger(__name__)


class RussianMozillaDataset(CustomAudioDataset):
    def __init__(self, data_dir, part, *args, **kwargs):
        data = []
        data_dir = Path(data_dir)
        tsv_file_path = str(data_dir / (part + ".tsv"))
        with open(tsv_file_path) as file:
            tsv_file = csv.reader(file, delimiter="\t")
            for line in tsv_file:
                audio_path = data_dir / "clips" / line[1]
                text = line[2]
                entry = {"path": str(audio_path),
                         "text": BaseTextEncoder.normalize_russian_text(
                             text.strip())}
                data.append(entry)
        super().__init__(data, *args, **kwargs)


if __name__ == "__main__":
    text_encoder = CTCCharTextEncoder.get_simple_alphabet()
    config_parser = ConfigParser.get_default_configs()

    ds = CustomDirAudioDataset("data/datasets/custom/audio",
                               text_encoder=text_encoder,
                               config_parser=config_parser)
    item = ds[0]
    print(item)
