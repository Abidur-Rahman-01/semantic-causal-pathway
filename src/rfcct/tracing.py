"""Contrastive decision-margin causal tracing and corruption calibration.

FCCT traces only present objects with p(Yes). Hallucination is a wrong "Yes"
for an absent object, which that probe never observes. R-FCCT keeps FCCT's
three runs (clean, Gaussian-corrupted, patched) and its restoration operator,
and adds:

* the correctness margin ``s = y * (logit_Yes - logit_No)`` with
  ``y = +1`` for present and ``y = -1`` for absent objects, so both kinds of
  error are traced with one quantity;
* the paper's "measurable drop" condition made explicit: a trace is used only
  when the clean answer is correct (``s_clean > 0``) and corruption lowers the
  margin by at least ``min_margin_gap``;
* p(Yes) is recorded too, so FCCT's probability RR (Eq. 2) can be computed on
  exactly the same runs.

Only the last token's MHSA and MLP outputs are patched: those are the only
activations IRI injects.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from PIL import Image

from fcct.core import capture, patch_component
from fcct.io import fingerprint
from fcct.llava import gaussian_corruption, object_patch_indices

COMPONENTS = ("attention", "mlp")


def resolve_token(tokenizer, text, expected):
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) != 1 or ids[0] != expected:
        raise ValueError(f"{text!r} must encode to the single token {expected}, got {ids}")
    return ids[0]


@torch.inference_mode()
def score(model, batch, yes_id, no_id):
    """Next-token decision statistics at the final prompt position."""
    logits = model(**batch, use_cache=False, logits_to_keep=1).logits[0, -1].float()
    probs = torch.softmax(logits, dim=-1)
    return {"margin": float(logits[yes_id] - logits[no_id]), "p_yes": float(probs[yes_id]),
            "p_no": float(probs[no_id]), "top_id": int(torch.argmax(logits))}


def object_visible(backend, image, row, batch):
    """Same crop-visibility rule as the FCCT reconstruction (present objects only)."""
    height, width = batch["pixel_values"].shape[-2:]
    grid = height // backend.patch_size, width // backend.patch_size
    return bool(object_patch_indices(image, row["boxes"], backend.processor.image_processor,
                                     backend.patch_size, grid))


def contrastive_trace_sample(backend, image, row, *, yes_id, no_id, sigma, seed, min_margin_gap,
                             check_visibility=True):
    sign = {"yes": 1.0, "no": -1.0}[row["label"]]
    batch, _ = backend.batch(image, row["question"])
    base = {"sample_id": row["sample_id"], "image_id": row["image_id"], "label": row["label"],
            "strategy": row.get("strategy"), "sigma": sigma}
    if check_visibility and row["label"] == "yes" and row.get("boxes") is not None:
        if not object_visible(backend, image, row, batch):
            return {**base, "status": "excluded_object_outside_crop"}
    noise_seed = int(fingerprint([seed, row["sample_id"], "rfcct"])[:16], 16)
    noisy = gaussian_corruption(image, sigma, noise_seed)
    corrupted, _ = backend.batch(noisy, row["question"])
    if not torch.equal(batch["input_ids"], corrupted["input_ids"]):
        raise ValueError("Clean/corrupted token alignment differs")
    last = batch["input_ids"].shape[1] - 1
    clean_embeds = backend.trace_batch(batch)
    corrupted_embeds = backend.trace_batch(corrupted)
    with capture(backend.layers, {"last": [last]}) as (cache, _positions):
        clean = score(backend.model, clean_embeds, yes_id, no_id)
    corrupt = score(backend.model, corrupted_embeds, yes_id, no_id)
    s_clean, s_corrupted = sign * clean["margin"], sign * corrupt["margin"]
    result = {**base, "noise_seed": noise_seed, "s_clean": s_clean, "s_corrupted": s_corrupted,
              "gap": s_clean - s_corrupted, "p_yes_clean": clean["p_yes"],
              "p_yes_corrupted": corrupt["p_yes"], "clean_top_id": clean["top_id"]}
    if s_clean <= 0:
        return {**result, "status": "excluded_clean_incorrect"}
    if not s_clean - s_corrupted >= min_margin_gap:
        return {**result, "status": "excluded_no_measurable_drop"}
    s_patched = {c: [] for c in COMPONENTS}
    p_patched = {c: [] for c in COMPONENTS}
    for index, layer in enumerate(backend.layers):
        for component in COMPONENTS:
            with patch_component(layer, component, [last], cache[index, component]):
                patched = score(backend.model, corrupted_embeds, yes_id, no_id)
            s_patched[component].append(sign * patched["margin"])
            p_patched[component].append(patched["p_yes"])
    if not all(math.isfinite(v) for c in COMPONENTS for v in s_patched[c]):
        raise ValueError("Non-finite patched margin")
    return {**result, "status": "complete", "s_patched": s_patched, "p_yes_patched": p_patched}


def calibrate_noise(backend, rows, root, *, yes_id, no_id, sigmas, seed, min_drop_rate, min_median_gap):
    """Pick the smallest sigma that induces a measurable drop on most questions.

    Criterion (fixed in the config before any tracing): among questions the
    model answers correctly on the clean image, at least ``min_drop_rate`` must
    have a positive margin drop and the median drop must be at least
    ``min_median_gap`` logits. If no sigma qualifies, the largest is used and
    this is reported.
    """
    clean_margins, table = {}, []
    for sigma in sorted(sigmas):
        gaps = []
        for row in rows:
            sign = {"yes": 1.0, "no": -1.0}[row["label"]]
            with Image.open(root / row["image"]) as source:
                image = source.convert("RGB")
            if row["sample_id"] not in clean_margins:
                batch, _ = backend.batch(image, row["question"])
                clean_margins[row["sample_id"]] = sign * score(backend.model, batch, yes_id, no_id)["margin"]
            s_clean = clean_margins[row["sample_id"]]
            if s_clean <= 0:
                continue
            noise_seed = int(fingerprint([seed, row["sample_id"], "rfcct"])[:16], 16)
            batch, _ = backend.batch(gaussian_corruption(image, sigma, noise_seed), row["question"])
            gaps.append(s_clean - sign * score(backend.model, batch, yes_id, no_id)["margin"])
        gaps = np.asarray(gaps)
        entry = {"sigma": sigma, "clean_correct": int(len(gaps)),
                 "drop_rate": float((gaps > 0).mean()) if len(gaps) else 0.0,
                 "median_gap": float(np.median(gaps)) if len(gaps) else 0.0}
        entry["qualifies"] = entry["drop_rate"] >= min_drop_rate and entry["median_gap"] >= min_median_gap
        table.append(entry)
        print(f"Noise sigma={sigma}: drop_rate={entry['drop_rate']:.3f} median_gap={entry['median_gap']:.3f}", flush=True)
    chosen = next((e for e in table if e["qualifies"]), None)
    return {"chosen_sigma": (chosen or table[-1])["sigma"], "criterion_met": chosen is not None,
            "criterion": {"min_drop_rate": min_drop_rate, "min_median_gap": min_median_gap},
            "questions": len(rows), "table": table}
