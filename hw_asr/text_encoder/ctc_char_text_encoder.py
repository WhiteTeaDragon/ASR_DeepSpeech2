from typing import List, Tuple
from transformers import TransfoXLTokenizer, TransfoXLLMHeadModel
from tqdm import tqdm
from collections import defaultdict

import torch

from hw_asr.text_encoder.char_text_encoder import CharTextEncoder


def _truncate_beam(paths, beam_size):
    return dict(sorted(paths.items(), key=lambda x: x[1])[-beam_size:])


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
        self.device = None
        self.tokenizer = None
        self.model = None

    def _extend_and_merge(self, next_char_probs, src_paths, alpha, beta):
        new_paths = defaultdict(float)
        for next_char_ind in range(len(next_char_probs)):
            next_char_prob = next_char_probs[next_char_ind]
            next_char = self.ind2char[next_char_ind]
            for (text, last_char), path_prob in src_paths.items():
                new_prefix = text
                if next_char != last_char and next_char != self.EMPTY_TOK:
                    new_prefix += next_char
                new_paths[
                    (new_prefix, next_char)] += path_prob * next_char_prob
        res_dict = {}
        for (new_prefix, _), value in new_paths.items():
            inputs = self.tokenizer(new_prefix, return_tensors="pt")
            outputs = self.model(**inputs, labels=inputs["input_ids"])
            logits = outputs.logits
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            target_log_probs = log_probs[:, :-1].gather(
                2, inputs[:, 1:].unsqueeze(2)).squeeze(2)
            neural_lm_score = torch.sum(target_log_probs, dim=-1)
            res_dict[new_prefix] = value + alpha * torch.exp(neural_lm_score) \
                + beta * len(new_prefix)
        return res_dict

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

    def ctc_beam_search(self, probs: torch.tensor, alpha=1, beta=0.5,
                        beam_size: int = 100, device=None) -> \
            List[Tuple[str, float]]:
        """
        Performs beam search and returns a list of pairs (hypothesis,
        hypothesis probability).
        """
        assert len(probs.shape) == 2
        char_length, voc_size = probs.shape
        assert voc_size == len(self.ind2char)
        if self.tokenizer is None:
            self.tokenizer = TransfoXLTokenizer.from_pretrained('transfo-xl'
                                                                '-wt103')
        if self.model is None:
            self.model = TransfoXLLMHeadModel.from_pretrained('transfo-xl'
                                                              '-wt103')
        if self.device != device:
            self.device = device
            self.model = self.model.to(device)
        self.model.eval()
        paths = {('', self.EMPTY_TOK): 1.0}
        for next_char_probs in tqdm(probs):
            paths = self._extend_and_merge(next_char_probs, paths, alpha=alpha,
                                           beta=beta)
            paths = _truncate_beam(paths, beam_size)
        hypos = [(prefix, score) for (prefix, _), score in paths.items()]
        return sorted(hypos, key=lambda x: x[1], reverse=True)
