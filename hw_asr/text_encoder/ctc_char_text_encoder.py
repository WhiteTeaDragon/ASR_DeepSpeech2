from typing import List, Tuple

import torch

from hw_asr.text_encoder.char_text_encoder import CharTextEncoder


class CTCCharTextEncoder(CharTextEncoder):
    EMPTY_TOK = "^"

    def __init__(self, alphabet: List[str]):
        super().__init__(alphabet)
        self.ind2char = {
            0: self.EMPTY_TOK
        }
        for text in alphabet:
            self.ind2char[max(self.ind2char.keys()) + 1] = text
        self.char2ind = {v: k for k, v in self.ind2char.items()}

    def ctc_decode(self, inds: List[int]) -> str:
        ans = []
        for i in range(len(inds)):
            curr_token = self.ind2char[inds[i]]
            if (len(ans) == 0 or curr_token != ans[-1]) and curr_token != \
                    self.EMPTY_TOK:
                ans.append(curr_token)
        return ''.join(ans)

    def ctc_beam_search(self, probs: torch.tensor, beam_size: int = 100) -> \
            List[Tuple[str, float]]:
        """
        Performs beam search and returns a list of pairs (hypothesis,
        hypothesis probability).
        """
        assert len(probs.shape) == 2
        char_length, voc_size = probs.shape
        assert voc_size == len(self.ind2char)
        hypos = []
        # TODO: your code here
        raise NotImplementedError
        return sorted(hypos, key=lambda x: x[1], reverse=True)
