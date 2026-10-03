"""External evidence interventions and answer-distribution effects."""
from __future__ import annotations

import math
from typing import Callable

from PIL import Image, ImageFilter

from .metrics import js_divergence


def replace_mask(image: Image.Image, mask, method: str = "blur", radius: float = 22) -> Image.Image:
    """Replace mask pixels while preserving the original image canvas."""
    import numpy as np
    from PIL import Image as PILImage
    mask_img = mask.convert("L") if isinstance(mask, PILImage.Image) else PILImage.fromarray((np.asarray(mask) > 0).astype("uint8") * 255, mode="L")
    if mask_img.size != image.size:
        mask_img = mask_img.resize(image.size, PILImage.Resampling.NEAREST)
    original = image.convert("RGB")
    if method == "blur":
        replacement = original.filter(ImageFilter.GaussianBlur(radius))
    elif method == "gray":
        replacement = PILImage.new("RGB", original.size, (127, 127, 127))
    elif method == "inpaint":
        try:
            import cv2
            import numpy as np
        except ImportError as exc:
            raise ImportError("Install opencv-python-headless to use method='inpaint'.") from exc
        image_array = np.asarray(original)
        mask_array = np.asarray(mask_img)
        repaired = cv2.inpaint(cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR), mask_array, 3, cv2.INPAINT_TELEA)
        replacement = PILImage.fromarray(cv2.cvtColor(repaired, cv2.COLOR_BGR2RGB))
    else:
        raise ValueError("method must be 'blur', 'gray', or 'inpaint'")
    return PILImage.composite(replacement, original, mask_img)


def output_distribution(log_probabilities: dict[str, float], temperature: float = 1.0) -> dict[str, float]:
    """Normalize candidate-answer log probabilities over a fixed candidate set."""
    if not log_probabilities:
        raise ValueError("At least one candidate answer is required")
    temperature = max(float(temperature), 1e-8)
    peak = max(log_probabilities.values())
    weights = {k: math.exp((v - peak) / temperature) for k, v in log_probabilities.items()}
    total = sum(weights.values())
    return {k: v / total for k, v in weights.items()}


def external_effects(image: Image.Image, question: str, evidence_mask, candidate_answers: list[str],
                     answer_logprobs: Callable[[Image.Image, str, list[str]], dict[str, float]],
                     method: str = "blur", control_masks: list | None = None) -> dict:
    """Measure necessity and sufficiency on a fixed answer set.

    `answer_logprobs` must use the same prompt, candidate set, and model settings in every run.
    Necessity compares original to evidence-deleted image; sufficiency compares original to
    evidence-only image. Control deletions help quantify generic masking sensitivity.
    """
    from PIL import Image as PILImage
    mask_img = evidence_mask.convert("L") if isinstance(evidence_mask, PILImage.Image) else PILImage.fromarray((__import__('numpy').asarray(evidence_mask) > 0).astype('uint8') * 255, mode='L')
    if mask_img.size != image.size:
        mask_img = mask_img.resize(image.size, PILImage.Resampling.NEAREST)
    full_logs = answer_logprobs(image, question, candidate_answers)
    p_full = output_distribution(full_logs)
    deleted = replace_mask(image, mask_img, method=method)
    p_deleted = output_distribution(answer_logprobs(deleted, question, candidate_answers))
    evidence_only = PILImage.composite(image.convert("RGB"), PILImage.new("RGB", image.size, (127,127,127)), mask_img)
    p_only = output_distribution(answer_logprobs(evidence_only, question, candidate_answers))
    controls = []
    for control in control_masks or []:
        p_control = output_distribution(answer_logprobs(replace_mask(image, control, method=method), question, candidate_answers))
        controls.append(js_divergence(p_full, p_control))
    necessity = js_divergence(p_full, p_deleted)
    control_mean = sum(controls) / len(controls) if controls else None
    control_std = (math.sqrt(sum((value - control_mean) ** 2 for value in controls) / len(controls))
                   if controls else None)
    return {
        "candidate_answers": candidate_answers,
        "p_full": p_full,
        "p_deleted": p_deleted,
        "p_evidence_only": p_only,
        "necessity_js": necessity,
        "necessity_js_control_adjusted": necessity - control_mean if control_mean is not None else None,
        "sufficiency_similarity": 1 - min(1.0, js_divergence(p_full, p_only) / math.log(2)),
        "control_deletion_js": controls,
        "control_deletion_js_mean": control_mean,
        "control_deletion_js_std": control_std,
        "intervention_method": method,
        "claim_boundary": "behavioral output-distribution sensitivity; not semantic truth",
    }
