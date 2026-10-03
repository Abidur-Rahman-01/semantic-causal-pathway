"""Deterministic area/shape-matched spatial control masks."""
from __future__ import annotations

import numpy as np


def sample_translated_control_masks(target_mask, n: int = 5, seed: int = 17,
                                   max_iou: float = 0.05, max_attempts: int = 2000) -> list[np.ndarray]:
    """Translate the exact binary mask to non-overlapping valid locations in-frame.

    Controls preserve mask shape and pixel area. They are spatial controls, not guaranteed
    semantically irrelevant; manually inspect them and exclude other critical evidence.
    """
    target = np.asarray(target_mask).astype(bool)
    if target.ndim != 2 or not target.any():
        raise ValueError("target_mask must be a nonempty 2D mask")
    ys, xs = np.where(target)
    y0, y1, x0, x1 = ys.min(), ys.max()+1, xs.min(), xs.max()+1
    patch = target[y0:y1, x0:x1]
    height, width = target.shape
    if patch.shape[0] > height or patch.shape[1] > width:
        raise ValueError("mask bounds exceed frame")
    rng = np.random.default_rng(seed)
    controls, attempts = [], 0
    while len(controls) < n and attempts < max_attempts:
        attempts += 1
        top = int(rng.integers(0, height-patch.shape[0]+1))
        left = int(rng.integers(0, width-patch.shape[1]+1))
        candidate = np.zeros_like(target)
        candidate[top:top+patch.shape[0], left:left+patch.shape[1]] = patch
        intersection = np.logical_and(candidate, target).sum()
        union = np.logical_or(candidate, target).sum()
        iou = intersection / union if union else 0
        if iou > max_iou or any(np.array_equal(candidate, prior) for prior in controls):
            continue
        controls.append(candidate)
    if len(controls) < n:
        raise RuntimeError(f"Generated only {len(controls)}/{n} spatial controls. Use a smaller mask or fewer controls.")
    return controls
