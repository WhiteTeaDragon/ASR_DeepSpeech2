import logging
from typing import List
import torch

logger = logging.getLogger(__name__)


def collate_fn(batch_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """
    x, y = [], []
    input_specs_lengths, output_text_lengths = [], []
    texts = []
    audios = []
    sample_rates = []
    for i in range(len(batch_items)):
        x.append(batch_items[i]["spectrogram"].squeeze(0).t())
        y.append(batch_items[i]["text_encoded"].squeeze(0))
        input_specs_lengths.append(x[-1].shape[0])
        output_text_lengths.append(y[-1].shape[0])
        texts.append(batch_items[i]["text"])
        audios.append(batch_items[i]["audio"])
        sample_rates.append(batch_items[i]["sample_rate"])
    x = torch.nn.utils.rnn.pad_sequence(x, batch_first=True, padding_value=0.)
    y = torch.nn.utils.rnn.pad_sequence(y, batch_first=True)
    return {"spectrogram": x, "text_encoded": y, "spectrogram_length":
            torch.tensor(input_specs_lengths, dtype=torch.int32),
            "text_encoded_length": torch.tensor(output_text_lengths,
                                                dtype=torch.int32), "text":
            texts, "audio": audios, "sample_rate": sample_rates}
