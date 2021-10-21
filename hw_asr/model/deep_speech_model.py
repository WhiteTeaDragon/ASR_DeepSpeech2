import math
from torch import nn

from hw_asr.base import BaseModel


class DeepSpeechModel(BaseModel):
    def __init__(self, n_feats, n_class, hidden_size, n_layers, dropout,
                 *args, **kwargs):
        super().__init__(n_feats, n_class, *args, **kwargs)
        self.convolutional = nn.Sequential(
            nn.Conv2d(1, 32, (41, 11), stride=(2, 2), padding=(20, 5)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, (21, 11), stride=(2, 1), padding=(10, 5)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 96, (21, 11), stride=(2, 1), padding=(10, 5)),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        rnn_input_size = int(math.floor(n_feats) / 2) + 1
        rnn_input_size = int(math.floor(rnn_input_size + 2 * 20 - 41) / 2 + 1)
        rnn_input_size = int(math.floor(rnn_input_size + 2 * 10 - 21) / 2 + 1)
        rnn_input_size = int(math.floor(rnn_input_size + 2 * 10 - 21) / 2 + 1)
        rnn_input_size *= 96
        self.lstm = nn.LSTM(input_size=rnn_input_size, hidden_size=hidden_size,
                            num_layers=n_layers, batch_first=False,
                            dropout=dropout, bidirectional=True)
        self.fully_connected = nn.Sequential(
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, n_class)
        )

    def forward(self, spectrogram, *args, **kwargs):
        packed_input = nn.utils.rnn.pack_padded_sequence(
            spectrogram, kwargs["spectrogram_length"], batch_first=True,
            enforce_sorted=False)
        x = self.convolutional(packed_input)
        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[2],
                   sizes[3])  # Collapse feature dimension
        x = x.transpose(1, 2).transpose(0, 1).contiguous()  # TxNxH
        lstm_out, _ = self.lstm(x)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out,
                                                       batch_first=False)
        x = self.fully_connected(lstm_out)
        x = x.transpose(0, 1)
        return {"logits": x}

    def transform_input_lengths(self, input_lengths):
        return input_lengths
