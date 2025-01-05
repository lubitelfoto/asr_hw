# Based on seminar materials
from jiwer import cer, wer

# Don't forget to support cases when target_text == ''


def calc_cer(target_text, predicted_text) -> float:
    if not target_text:
        return float(len(predicted_text.split()))
    return cer(target_text, predicted_text)


def calc_wer(target_text, predicted_text) -> float:
    if not target_text:
        return float(len(predicted_text.split()))
    return wer(target_text, predicted_text)
