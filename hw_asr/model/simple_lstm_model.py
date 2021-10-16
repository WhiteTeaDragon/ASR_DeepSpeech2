from torch import nn

from hw_asr.base import BaseModel


class SimpleLSTMModel(BaseModel):
    def __init__(self, n_feats, n_class, n_layers, hidden_size=512,
                 dropout=0.25, *args, **kwargs):
        super().__init__(n_feats, n_class, *args, **kwargs)
        self.LSTM = nn.LSTM(input_size=n_feats, hidden_size=hidden_size,
                                num_layers=n_layers, batch_first=True,
                                dropout=dropout, bidirectional=True)
        self.linear = nn.Linear(in_features=2 * hidden_size,
                                out_features=n_class)

    def forward(self, spectrogram, *args, **kwargs):
        packed_input = nn.utils.rnn.pack_padded_sequence(
            spectrogram, kwargs["spectrogram_length"], batch_first=True,
            enforce_sorted=False)
        lstm_out, _ = self.LSTM(packed_input)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out,
                                                       batch_first=True)
        return {"logits": self.linear(lstm_out)}

    def transform_input_lengths(self, input_lengths):
        return input_lengths  # we don't reduce time dimension here
