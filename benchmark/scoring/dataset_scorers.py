"""Unified dataset scorer interface.

Dispatches to dataset-specific scoring rules and tracks the exact scorer version:
  - VQAv2: vqav2_soft_consensus_v2.0
  - GQA:   gqa_exact_match_v1.2
"""
from __future__ import annotations

from .vqa_scorer import vqa_consensus_score, normalize_vqa_answer
from .gqa_scorer import gqa_exact_score, normalize_gqa_answer

SCORER_VERSIONS = {
    "vqav2": "vqav2_soft_consensus_v2.0",
    "gqa": "gqa_exact_match_v1.2",
    "default": "dataset_answer_score_v1.0",
}


def score_prediction(
    prediction: str,
    answers: list[str] | None,
    dataset: str | None,
    threshold: float = 0.5,
) -> tuple[float | None, bool, str]:
    """Score a model prediction against gold reference answers.
    
    Returns:
        (score, is_correct, scorer_version)
    """
    if not answers:
        return None, False, SCORER_VERSIONS["default"]

    ds_key = str(dataset or "").strip().casefold()
    if "gqa" in ds_key:
        score = gqa_exact_score(prediction, answers)
        version = SCORER_VERSIONS["gqa"]
    else:
        # Default / VQAv2 consensus rule
        score = vqa_consensus_score(prediction, answers)
        version = SCORER_VERSIONS["vqav2"]

    is_correct = score is not None and score >= threshold
    return score, is_correct, version


def normalize_answer(answer: str, dataset: str | None) -> str:
    """Normalize answer text based on the target dataset."""
    ds_key = str(dataset or "").strip().casefold()
    if "gqa" in ds_key:
        return normalize_gqa_answer(answer)
    return normalize_vqa_answer(answer)
