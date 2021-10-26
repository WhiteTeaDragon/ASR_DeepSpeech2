import editdistance


def calc_cer(target_text, predicted_text) -> float:
    if len(target_text) == 0 and len(predicted_text) == 0:
        return 1
    if len(target_text) == 0:
        return 0
    return editdistance.eval(target_text, predicted_text) / len(target_text)


def calc_wer(target_text, predicted_text) -> float:
    reference = target_text.split()
    prediction = predicted_text.split()
    if len(reference) == 0 and len(prediction) == 0:
        return 1
    if len(reference) == 0:
        return 0
    return editdistance.eval(reference, predicted_text.split()) / len(reference
                                                                      )
