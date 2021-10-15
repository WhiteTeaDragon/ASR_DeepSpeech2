from torch import nn

from hw_asr.base import BaseModel


class SimpleLSTMModel(BaseModel):
    def __init__(self, n_feats, n_class, n_layers, hidden_size=512, *args,
                 **kwargs):
        super().__init__(n_feats, n_class, *args, **kwargs)
        self.LSTM = nn.LSTM(input_size=n_feats, hidden_size=hidden_size,
                                num_layers=n_layers, batch_first=True,
                                dropout=0.25, bidirectional=True)
        self.linear = nn.Linear(in_features=hidden_size, out_features=n_class)

    def forward(self, spectrogram, *args, **kwargs):
        after_lstm = self.LSTM(spectrogram)
        return {"logits": self.linear(after_lstm)}

    def transform_input_lengths(self, input_lengths):
        return input_lengths  # we don't reduce time dimension here
