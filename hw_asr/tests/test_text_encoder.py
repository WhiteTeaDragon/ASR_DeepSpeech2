import unittest
import torch

from hw_asr.text_encoder.ctc_char_text_encoder import CTCCharTextEncoder


class TestTextEncoder(unittest.TestCase):
    def test_ctc_decode(self):
        text_encoder = CTCCharTextEncoder.get_simple_alphabet()
        text = "i^^ ^w^i^sss^hhh^   i ^^^s^t^aaaar^teee^d dddddd^oooo^in^g " \
               "tttttttth^iiiis h^^^^^^^^w^ e^a^r^li^er "
        true_text = "i wish i started doing this hw earlier"
        inds = [text_encoder.char2ind[c] for c in text]
        decoded_text = text_encoder.ctc_decode(inds)
        self.assertIn(decoded_text, true_text)

    def test_beam_search(self):
        vocab = list("ab-")
        probs = [[[0.2, 0, 0.8],
                 [0.4, 0, 0.6]]]  # (batch_size, timestamps, vocab_size)
        text_encoder = CTCCharTextEncoder(vocab)
        res = text_encoder.ctc_beam_search(probs, [2], 2)
        true_res = [("a", 0.52), ("", 0.48)]
        self.assertListEqual(res, true_res)
