import torch
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
            nn.BatchNorm2d(96),
            nn.ReLU()
        )
        rnn_input_size = 96 * n_feats // 8
        self.lstm = nn.LSTM(input_size=rnn_input_size, hidden_size=hidden_size,
                            num_layers=n_layers, batch_first=True,
                            dropout=dropout, bidirectional=True)
        self.bn = nn.BatchNorm1d(2 * hidden_size)
        self.fc = nn.Linear(2 * hidden_size, n_class)

    def forward(self, spectrogram, *args, **kwargs):
        print(spectrogram.shape)
        x = self.convolutional(torch.transpose(spectrogram, 1, 2).unsqueeze(1)
                               )
        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[2],
                   sizes[3])  # Collapse feature dimension
        x = x.transpose(1, 2)
        lstm_out, _ = self.lstm(x)
        lstm_out = torch.transpose(lstm_out, 1, 2)
        x = self.bn(lstm_out)
        x = torch.transpose(x, 1, 2)
        x = self.fc(x)
        return {"logits": x}

    def transform_input_lengths(self, input_lengths):
        return input_lengths // 2
