from typing import List, Union
from torch import Tensor

import youtokentome as yttm
import numpy as np

from hw_asr.text_encoder.ctc_char_text_encoder import CTCCharTextEncoder


class CTCBPETextEncoder(CTCCharTextEncoder):
    # EMPTY_TOK = "<PAD>"
    def __init__(self, bpe_object):
        self.bpe_object = bpe_object
        super().__init__(self.bpe_object.vocab())
        self.vocab = self.bpe_object.vocab()

    def decode(self, vector: Union[Tensor, np.ndarray, List[int]]):
        res = self.bpe_object.decode(vector)
        if len(res) == 0:
            return ""
        return res[0]

    def encode(self, text) -> Tensor:
        return Tensor(
            self.bpe_object.encode(
                text, output_type=yttm.OutputType.ID)).unsqueeze(0)

    def ctc_decode(self, inds: List[int]) -> str:
        ans = []
        for i in range(len(inds)):
            if len(ans) == 0 or inds[i] != ans[-1]:
                ans.append(inds[i])
        ans_final = []
        for i in range(len(ans)):
            if ans[i] != 0:
                ans_final.append(ans[i])
        res = self.bpe_object.decode(ans_final)
        if len(res) == 0:
            return ""
        return res[0]

    def __len__(self):
        return len(self.vocab)
