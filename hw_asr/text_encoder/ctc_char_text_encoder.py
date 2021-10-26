from typing import List, Tuple
from speechbrain.utils.data_utils import download_file
from pyctcdecode import build_ctcdecoder

import torch
import shutil
import math
import numpy as np
import gzip
import gdown

from hw_asr.text_encoder.char_text_encoder import CharTextEncoder
from hw_asr.utils import ROOT_PATH


class CTCCharTextEncoder(CharTextEncoder):
    EMPTY_TOK = "^"

    def __init__(self, alphabet: List[str], language="eng"):
        super().__init__(alphabet)
        self.ind2char = {
            0: self.EMPTY_TOK
        }
        for text in alphabet:
            self.ind2char[max(self.ind2char.keys()) + 1] = text
        self.char2ind = {v: k for k, v in self.ind2char.items()}
        self.file_path = None
        self.upper_file_path = None
        data_dir = ROOT_PATH / "lm_model"
        data_dir.mkdir(exist_ok=True, parents=True)
        self._data_dir = data_dir
        self.decoder = None
        vocab = list(self.ind2char.values())
        vocab[0] = ""
        self.vocab = vocab

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
                        alpha=0.5, beta=1, use_lm=True, lang="eng") -> \
            List[Tuple[str, float]]:
        """
        Performs beam search and returns a list of pairs (hypothesis,
        hypothesis probability).
        """
        assert lang in ("eng", "rus")
        assert len(probs.shape) == 2
        char_length, voc_size = probs.shape
        assert voc_size == len(self.ind2char)
        if self.file_path is None and use_lm:
            print(f"Loading kenlm")
            if lang == "eng":
                arch_path = self._data_dir / "3-gram.pruned.1e-7.arpa.gz"
                upper_file_path = self._data_dir / \
                                       "upper-3-gram.pruned.1e-7.arpa"
                self.file_path = self._data_dir / "3-gram.pruned.1e-7.arpa"
                download_file("http://www.openslr.org/resources/11/3-gram."
                              "pruned.1e-7.arpa.gz", arch_path)
                with gzip.open(arch_path, 'rb') as f_in:
                    with open(upper_file_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                with open(upper_file_path, "r") as f_in:
                    with open(self.file_path, "w") as f_out:
                        for line in f_in:
                            f_out.write(line.lower().replace("'", ""))
            if lang == "rus":
                url = "https://drive.google.com/file/d" \
                      "/1OeVsDTpM4lkWl9l_7eFVtxorCn_PE_aA "
                self.file_path = self._data_dir / "russian.arpa"
                gdown.download(url, self.file_path, quiet=False)
        if self.decoder is None and use_lm:
            self.decoder = build_ctcdecoder(self.vocab,
                                   str(self.file_path),
                                   alpha=alpha,  # tuned on a val set
                                   beta=beta,  # tuned on a val set
            )
        curr_decoder = self.decoder
        if not use_lm:
            decoder = build_ctcdecoder(self.vocab, None,
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
                                          beam_size,
                                          hotword_weight=0)
        for i in range(len(hypos)):
            hypos[i] = (hypos[i][0], math.exp(hypos[i][-1]))
        return sorted(hypos, key=lambda x: x[1], reverse=True)
