"""Official VQAv2 consensus scoring rule (v2.0).

Follows the standard VQA evaluation metric:
  accuracy = min(1.0, count(matching answers) / 3.0)
applied to 10 human reference answers after case-folding, punctuation removal,
article stripping, and number word normalization.
"""
from __future__ import annotations

import re
import string

NUMBER_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
}


def normalize_vqa_answer(value: str) -> str:
    """Normalize answer text according to standard VQA evaluation rules."""
    text = str(value or "").lower().strip()
    # Remove comma between digits (e.g., 1,000 -> 1000)
    text = re.sub(r"(?<=\d),(?=\d)", "", text)
    # Remove punctuation
    text = re.sub(r"[^\w\s']", " ", text)
    tokens = text.split()
    # Strip articles (a, an, the) and map number words
    filtered = [
        NUMBER_WORDS.get(tok, tok) for tok in tokens
        if tok not in {"a", "an", "the"}
    ]
    return " ".join(filtered)


def vqa_consensus_score(prediction: str, answers: list[str] | None) -> float | None:
    """Compute VQAv2 soft consensus score against reference answers.
    
    Returns float in [0.0, 1.0], or None if no reference answers are supplied.
    """
    if not answers:
        return None
    pred_norm = normalize_vqa_answer(prediction)
    refs_norm = [normalize_vqa_answer(ans) for ans in answers if str(ans).strip()]
    if not refs_norm:
        return 0.0
    matches = sum(ans == pred_norm for ans in refs_norm)
    return min(matches / 3.0, 1.0)
