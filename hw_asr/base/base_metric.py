from typing import List

from torch import Tensor


class BaseMetric:
    def __init__(self, name=None, *args, **kwargs):
        self.name = name if name is not None else type(self).__name__
        self.metric_function = None
        self.text_encoder = None

    def __call__(self, log_probs: Tensor, text: List[str], *args, **kwargs):
        argmax_inds = log_probs.cpu().argmax(-1)
        argmax_inds = [
            inds[: int(ind_len)]
            for inds, ind_len in zip(argmax_inds, kwargs["log_probs_length"])
        ]
        if kwargs["beam_search"]:
            predictions = []
            for i in range(len(log_probs)):
                ind_len = int(kwargs["log_probs_length"][i])
                predictions.append(self.text_encoder.ctc_beam_search(
                    log_probs[i, :ind_len])[0][0])
        else:
            if hasattr(self.text_encoder, "ctc_decode"):
                predictions = [self.text_encoder.ctc_decode(inds.tolist()) for
                               inds in argmax_inds]
            else:
                predictions = [self.text_encoder.decode(inds.tolist()) for
                               inds in argmax_inds]
        ers = []
        for pred_text, target_text in zip(predictions, text):
            ers.append(self.metric_function(target_text, pred_text))
        return sum(ers) / len(ers)
