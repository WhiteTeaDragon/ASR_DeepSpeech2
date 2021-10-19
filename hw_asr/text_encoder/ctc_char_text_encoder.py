from typing import List, Tuple
from speechbrain.utils.data_utils import download_file
from pyctcdecode import build_ctcdecoder

import torch
import shutil
import os
import gzip
import re
import math
import numpy as np

from hw_asr.text_encoder.char_text_encoder import CharTextEncoder
from hw_asr.utils import ROOT_PATH


def gunzip_something(gzipped_file_name, work_dir):
    """gunzip the given gzipped file"""

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
        self.decoder = None

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

    def ctc_beam_search(self, probs: torch.tensor, beam_size: int = 100,
                        alpha=0.5, beta=1, use_lm=True) -> \
            List[Tuple[str, float]]:
        """
        Performs beam search and returns a list of pairs (hypothesis,
        hypothesis probability).
        """
        assert len(probs.shape) == 2
        char_length, voc_size = probs.shape
        assert voc_size == len(self.ind2char)
        if self.file_path is None and use_lm:
            arch_path = self._data_dir / "3-gram.pruned.1e-7.arpa.gz"
            self.file_path = self._data_dir / "3-gram.pruned.1e-7.arpa"
            print(f"Loading kenlm")
            download_file("http://www.openslr.org/resources/11/3-gram.pruned"
                          ".1e-7.arpa.gz", arch_path)
            shutil.unpack_archive(arch_path, self._data_dir, "gz")
        if self.decoder is None and use_lm:
            vocab = list(self.ind2char.values())
            vocab[0] = ""
            self.decoder = build_ctcdecoder(vocab,
                                   str(self.file_path),
                                   alpha=alpha,  # tuned on a val set
                                   beta=beta,  # tuned on a val set
            )
        curr_decoder = self.decoder
        if not use_lm:
            vocab = list(self.ind2char.values())
            vocab[0] = ""
            decoder = build_ctcdecoder(vocab, None,
                                            alpha=alpha,  # tuned on a val set
                                            beta=beta)  # tuned on a val set
            curr_decoder = decoder
        # hypos = self.decoder.decode_beams(torch.cat((probs.detach().cpu(),
        #                                         torch.zeros(char_length,
        #                                                     1).detach()),
        #                                        1).numpy(), beam_size)
        log_probs = np.log(np.clip(probs.detach().cpu().numpy(),
                                   1e-15, 1))
        hypos = curr_decoder.decode_beams(log_probs,
                                          beam_size, token_min_logp=-10000,
                                          beam_prune_logp=-10000,
                                          hotword_weight=0)
        for i in range(len(hypos)):
            hypos[i] = (hypos[i][0], math.exp(hypos[i][-1]))
        return sorted(hypos, key=lambda x: x[1], reverse=True)
