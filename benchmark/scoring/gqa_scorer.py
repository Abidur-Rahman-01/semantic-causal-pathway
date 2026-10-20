"""Official GQA scoring rule (v1.2).

Follows standard GQA exact-match evaluation on normalized answers:
lower-case, punctuation-stripped, and article-free string comparison.
Also supports substring/word-boundary matching for conversational VLM generation.
"""
from __future__ import annotations

import re
import string


def normalize_gqa_answer(value: str) -> str:
    """Normalize answer string according to GQA rules."""
    text = str(value or "").lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def gqa_exact_score(prediction: str, answers: list[str] | None) -> float | None:
    """Compute GQA answer score (1.0 for match, 0.0 for non-match).
    
    Returns float in {0.0, 1.0}, or None if no reference answers are supplied.
    """
    if not answers:
        return None
    pred_norm = normalize_gqa_answer(prediction)
    refs_norm = [normalize_gqa_answer(ans) for ans in answers if str(ans).strip()]
    if not refs_norm:
        return 0.0
    for ref in refs_norm:
        if pred_norm == ref or re.search(r"\b" + re.escape(ref) + r"\b", pred_norm):
            return 1.0
    return 0.0
