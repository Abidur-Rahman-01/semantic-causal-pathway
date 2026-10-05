"""Deterministic nuisance transformations and human semantic-validity gates."""
from __future__ import annotations

import io

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


def generate_photometric_variants(image: Image.Image, question: str) -> list[dict]:
    """Generate mild variants; avoid brightness/contrast edits for color questions."""
    image = image.convert("RGB")
    low = question.lower()
    color_sensitive = any(term in low for term in ("color", "colour", "shade"))
    variants = []
    for quality in ((90, 92, 95, 98) if color_sensitive else (92, 98)):
        stream = io.BytesIO()
        image.save(stream, format="JPEG", quality=quality)
        jpeg = Image.open(io.BytesIO(stream.getvalue())).convert("RGB")
        variants.append({"transform": f"jpeg_quality_{quality}", "image": jpeg})
    if not color_sensitive:
        variants[:0] = [
            {"transform": "brightness_0.97", "image": ImageEnhance.Brightness(image).enhance(0.97)},
            {"transform": "contrast_1.03", "image": ImageEnhance.Contrast(image).enhance(1.03)},
        ]
    return variants


def _mask_image(mask, size: tuple[int, int]) -> Image.Image:
    if isinstance(mask, Image.Image):
        result = mask.convert("L")
    else:
        result = Image.fromarray((np.asarray(mask) > 0).astype("uint8") * 255, mode="L")
    if result.size != size:
        result = result.resize(size, Image.Resampling.NEAREST)
    return result


def generate_geometric_variants(image: Image.Image, fraction: float = 0.02,
                                evidence_mask=None) -> list[dict]:
    """Translate the whole image and its evidence mask without cropping or wrapping."""
    image = image.convert("RGB")
    dx, dy = max(1, round(image.width * fraction)), max(1, round(image.height * fraction))
    operations = [
        ("translate_left_2pct", (1, 0, dx, 0, 1, 0)),
        ("translate_right_2pct", (1, 0, -dx, 0, 1, 0)),
        ("translate_up_2pct", (1, 0, 0, 0, 1, dy)),
        ("translate_down_2pct", (1, 0, 0, 0, 1, -dy)),
    ]
    source_mask = _mask_image(evidence_mask, image.size) if evidence_mask is not None else None
    variants = []
    for name, matrix in operations:
        record = {"transform": name, "image": image.transform(
            image.size, Image.Transform.AFFINE, matrix,
            resample=Image.Resampling.BICUBIC, fillcolor=(127, 127, 127))}
        if source_mask is not None:
            record["evidence_mask"] = source_mask.transform(
                image.size, Image.Transform.AFFINE, matrix,
                resample=Image.Resampling.NEAREST, fillcolor=0)
        variants.append(record)
    return variants


def generate_background_variants(image: Image.Image, evidence_mask) -> list[dict]:
    """Blur or neutralize non-evidence context while leaving critical pixels unchanged."""
    image = image.convert("RGB")
    mask = _mask_image(evidence_mask, image.size)
    blurred = image.filter(ImageFilter.GaussianBlur(radius=max(8, min(image.size) * 0.04)))
    neutral = Image.new("RGB", image.size, (127, 127, 127))
    return [
        {"transform": "background_blur", "image": Image.composite(image, blurred, mask)},
        {"transform": "background_neutral", "image": Image.composite(image, neutral, mask)},
    ]


def generate_noncritical_removal_variants(image: Image.Image, evidence_mask,
                                           noncritical_masks: list[dict]) -> list[dict]:
    """Inpaint explicitly annotated noncritical objects outside the reviewed evidence mask.

    Each item is {name, mask}. Candidate validity still requires human review.
    """
    import cv2

    image = image.convert("RGB")
    evidence = np.asarray(_mask_image(evidence_mask, image.size)) > 0
    source = np.asarray(image)
    candidates = []
    for item in noncritical_masks:
        raw = np.asarray(_mask_image(item["mask"], image.size)) > 0
        target = raw & ~evidence
        if not target.any():
            continue
        repaired = cv2.inpaint(cv2.cvtColor(source, cv2.COLOR_RGB2BGR),
                               (target.astype("uint8") * 255), 3, cv2.INPAINT_TELEA)
        name = str(item.get("name") or "object").strip().lower().replace(" ", "_")
        candidates.append({"transform": f"remove_noncritical_{name}",
                           "image": Image.fromarray(cv2.cvtColor(repaired, cv2.COLOR_BGR2RGB))})
    return candidates


def generate_semantic_variants(image: Image.Image, question: str, evidence_mask=None,
                               noncritical_masks: list[dict] | None = None,
                               include_photometric: bool = True,
                               include_geometric: bool = True,
                               include_background: bool = True) -> list[dict]:
    """Generate the supported T1–T4 candidate families; every result is unapproved."""
    variants = []
    if include_photometric:
        variants.extend(generate_photometric_variants(image, question))
    if include_geometric:
        variants.extend(generate_geometric_variants(image, evidence_mask=evidence_mask))
    if evidence_mask is not None and include_background:
        variants.extend(generate_background_variants(image, evidence_mask))
    if evidence_mask is not None and noncritical_masks:
        variants.extend(generate_noncritical_removal_variants(image, evidence_mask, noncritical_masks))
    return variants


def validity_gate(variant_record: dict) -> tuple[bool, list[str]]:
    """Require explicit answer, evidence, relation, and audit confirmation fields."""
    failures = []
    for key in ("answer_preserved", "critical_evidence_preserved", "relations_preserved", "human_audited"):
        if variant_record.get(key) is not True:
            failures.append(key)
    if not str(variant_record.get("reviewer", "")).strip():
        failures.append("reviewer")
    if not str(variant_record.get("review_date", "")).strip():
        failures.append("review_date")
    return not failures, failures
