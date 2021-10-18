from typing import List, Tuple
from speechbrain.utils.data_utils import download_file
from pyctcdecode import build_ctcdecoder

import torch
import kenlm
import shutil
import os
import gzip
import re

from hw_asr.text_encoder.char_text_encoder import CharTextEncoder
from hw_asr.utils import ROOT_PATH


def gunzip_something(gzipped_file_name, work_dir):
    "gunzip the given gzipped file"

    # see warning about filename
    filename = os.path.split(gzipped_file_name)[-1]
    filename = re.sub(r"\.gz$", "", filename, flags=re.IGNORECASE)

    with gzip.open(gzipped_file_name, 'rb') as f_in:  # <<==========
        # extraction happens here
        with open(os.path.join(work_dir, filename), 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)


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
        self.file_path = None
        data_dir = ROOT_PATH / "lm_model"
        data_dir.mkdir(exist_ok=True, parents=True)
        self._data_dir = data_dir
        shutil.register_unpack_format('gz',
                                      ['.gz', ],
                                      gunzip_something)

    def ctc_decode(self, inds: List[int]) -> str:
        ans = []
        for i in range(len(inds)):
            curr_token = self.ind2char[inds[i]]
            if len(ans) == 0 or curr_token != ans[-1]:
                ans.append(curr_token)
        ans_final = []
        for i in range(len(ans)):
            if ans[i] != self.EMPTY_TOK:
                ans_final.append(ans[i])
        return ''.join(ans_final)

    def ctc_beam_search(self, probs: torch.tensor, alpha=0.5, beta=1,
                        beam_size: int = 100, device=None) -> \
            List[Tuple[str, float]]:
        """
        Performs beam search and returns a list of pairs (hypothesis,
        hypothesis probability).
        """
        assert len(probs.shape) == 2
        char_length, voc_size = probs.shape
        assert voc_size == len(self.ind2char)
        if self.file_path is None:
            arch_path = self._data_dir / "4gram_big.arpa.gz"
            self.file_path = self._data_dir / "4gram_big.arpa"
            print(f"Loading kenlm")
            download_file("https://kaldi-asr.org/models/5/4gram_big.arpa.gz",
                          arch_path)
            shutil.unpack_archive(arch_path, self._data_dir, "gz")
            os.remove(str(arch_path))
        decoder = build_ctcdecoder(list(self.char2ind.keys()),
                                   str(self.file_path),
                                   alpha=alpha,  # tuned on a val set
                                   beta=beta,  # tuned on a val set
        )
        hypos = decoder.decode_beams(torch.cat((probs.cpu(),
                                                torch.zeros(char_length, 1)),
                                               1), beam_size)
        for i in range(len(hypos)):
            hypos[i] = (hypos[0], hypos[-1])
        return sorted(hypos, key=lambda x: x[1], reverse=True)
