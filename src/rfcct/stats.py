"""Causal-score estimators, stability selection, and paired evaluation statistics.

All resampling is over *images* (clusters): every question about one image
stays in the same bootstrap draw, because questions about the same image are
not independent observations.
"""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

CLASSES = ("yes", "no")


# ---------------------------------------------------------------------------
# Causal scores from contrastive traces
# ---------------------------------------------------------------------------

def image_tables(records, component, num_layers):
    """Per-image, per-class sums of numerator and denominator effects.

    A completed record holds the label, the correctness-margin drop
    ``gap = s_clean - s_corrupted`` (> 0 by construction of the filter), and
    ``s_patched[component][l]``. The restoration effect is
    ``n_l = s_patched_l - s_corrupted``.

    Returns ``(image_ids, N, D, C)`` with shapes ``[I, 2, L]``, ``[I, 2]``,
    ``[I, 2]`` holding per-image *means* of n and d per class and the
    per-class question counts.
    """
    sums_n = defaultdict(lambda: np.zeros((len(CLASSES), num_layers)))
    sums_d = defaultdict(lambda: np.zeros(len(CLASSES)))
    counts = defaultdict(lambda: np.zeros(len(CLASSES)))
    for record in records:
        if record["status"] != "complete":
            continue
        c = CLASSES.index(record["label"])
        patched = np.asarray(record["s_patched"][component], dtype=float)
        if patched.shape != (num_layers,) or not np.isfinite(patched).all():
            raise ValueError("Contrastive trace has an invalid patched-margin vector")
        gap = float(record["gap"])
        if not math.isfinite(gap) or gap <= 0:
            raise ValueError("Completed contrastive traces must have a positive margin drop")
        sums_n[record["image_id"]][c] += patched - float(record["s_corrupted"])
        sums_d[record["image_id"]][c] += gap
        counts[record["image_id"]][c] += 1
    image_ids = sorted(counts)
    if not image_ids:
        raise ValueError("No completed contrastive traces")
    C = np.stack([counts[i] for i in image_ids])
    safe = np.where(C > 0, C, 1.0)
    N = np.stack([sums_n[i] for i in image_ids]) / safe[:, :, None]
    D = np.stack([sums_d[i] for i in image_ids]) / safe
    return image_ids, N, D, C


def ratio_scores(N, D, C, weights=None, classes=CLASSES, combine="mean"):
    """Image-balanced ratio of means, per class, then combined over classes.

    ``beta_{c,l} = sum_i w_i [C_ic>0] N_icl / sum_i w_i [C_ic>0] D_ic``.
    Because every retained question has a positive drop, the denominator is a
    sum of positive terms and cannot approach zero through sign cancellation.
    ``weights`` are bootstrap multiplicities (``None`` = all ones).
    """
    if combine not in ("mean", "min"):
        raise ValueError("combine must be mean or min")
    w = np.ones(N.shape[0]) if weights is None else np.asarray(weights, dtype=float)
    per_class = []
    for name in classes:
        c = CLASSES.index(name)
        present = (C[:, c] > 0) * w
        denominator = float(present @ D[:, c])
        if denominator <= 0:
            return None
        per_class.append((present @ N[:, c, :]) / denominator)
    stacked = np.stack(per_class)
    return stacked.mean(0) if combine == "mean" else stacked.min(0)


def bootstrap_scores(N, D, C, *, draws, seed, classes=CLASSES, combine="mean"):
    """Image-cluster bootstrap distribution of the combined ratio score."""
    if ratio_scores(N, D, C, None, classes, combine) is None:
        raise ValueError(f"No completed traces for one of the classes {classes}")
    rng = np.random.default_rng(seed)
    images = N.shape[0]
    out, attempts = [], 0
    while len(out) < draws:
        attempts += 1
        if attempts > 100 * draws:
            raise ValueError("Bootstrap draws rarely contain every class; too few traced images")
        weights = np.bincount(rng.integers(images, size=images), minlength=images)
        score = ratio_scores(N, D, C, weights, classes, combine)
        if score is not None:  # a draw without one class is redrawn
            out.append(score)
    return np.stack(out)


def paper_rr_profile(records, component, num_layers, min_probability_gap):
    """FCCT Equation 2 on present objects with a measurable probability drop.

    This is the paper-faithful estimator: mean over images of mean RR, with RR
    computed in probability space for ``label == yes`` questions whose
    ``p_clean - p_corrupted > min_probability_gap`` ("induce measurable drops
    in output probabilities for existing objects").
    """
    by_image = defaultdict(list)
    for record in records:
        if record["status"] != "complete" or record["label"] != "yes":
            continue
        gap = record["p_yes_clean"] - record["p_yes_corrupted"]
        if gap <= min_probability_gap:
            continue
        patched = np.asarray(record["p_yes_patched"][component], dtype=float)
        by_image[record["image_id"]].append((patched - record["p_yes_corrupted"]) / gap)
    if not by_image:
        raise ValueError("No present-object traces pass the probability-drop filter")
    image_means = np.stack([np.mean(v, axis=0) for v in by_image.values()])
    if image_means.shape[1] != num_layers:
        raise ValueError("Layer count mismatch in paper RR profile")
    return image_means.mean(0), len(by_image)


# ---------------------------------------------------------------------------
# Stable, reachable source/target plans
# ---------------------------------------------------------------------------

def inclusion_frequency(boot, k):
    """Fraction of bootstrap draws in which each layer ranks in the top k."""
    top = np.argsort(-boot, axis=1, kind="stable")[:, :k]
    layers = boot.shape[1]
    return np.array([(top == l).any(axis=1).mean() for l in range(layers)])


def _order(layers, primary, score):
    return sorted(layers, key=lambda l: (-primary[l], -score[l], l))


def reachable_plan(score, k_source, k_target, *, priority_source=None, priority_target=None,
                   weights=None, eligible_sources=None, reachable=True, max_iter=64):
    """Select sources and targets so that every source has a later target.

    ``priority_*`` order candidates (stability frequencies, or the score
    itself). ``eligible_sources`` restricts source candidates (for example to
    layers whose lower confidence bound is positive). With ``reachable=False``
    the FCCT/IRI rule is reproduced: independent top-k sources and targets.
    """
    score = np.asarray(score, dtype=float)
    layers = len(score)
    ps = score if priority_source is None else np.asarray(priority_source, dtype=float)
    pt = score if priority_target is None else np.asarray(priority_target, dtype=float)
    candidates = list(range(layers)) if eligible_sources is None else sorted(eligible_sources)
    w = score if weights is None else np.asarray(weights, dtype=float)
    if not reachable:
        sources = sorted(_order(candidates, ps, score)[:k_source])
        targets = sorted(_order(range(layers), pt, score)[:k_target])
        return {"sources": sources, "targets": targets,
                "weights": {str(s): float(w[s]) for s in sources}}
    sources, targets = [], []
    limit = layers - 1  # a source needs some strictly later layer
    for _ in range(max_iter):
        pool = [l for l in candidates if l < limit]
        new_sources = sorted(_order(pool, ps, score)[:k_source])
        if not new_sources:
            return {"sources": [], "targets": [], "weights": {}}
        floor = min(new_sources)
        new_targets = sorted(_order([l for l in range(layers) if l > floor], pt, score)[:k_target])
        limit = max(new_targets)
        if new_sources == sources and new_targets == targets:
            break
        sources, targets = new_sources, new_targets
    # Drop any source that still cannot reach a target (cannot inject).
    sources = [s for s in sources if any(t > s for t in targets)]
    return {"sources": sources, "targets": targets,
            "weights": {str(s): float(w[s]) for s in sources}}


def stable_plan(score, boot, k_source, k_target, *, alpha=0.05, reachable=True):
    """R-FCCT selection: stability frequencies, positive LCB, LCB weights."""
    lcb = np.quantile(boot, alpha, axis=0)
    eligible = [l for l in range(len(score)) if lcb[l] > 0]
    plan = reachable_plan(score, k_source, k_target,
                          priority_source=inclusion_frequency(boot, k_source),
                          priority_target=inclusion_frequency(boot, k_target),
                          weights=lcb, eligible_sources=eligible, reachable=reachable)
    plan["lcb"] = [float(v) for v in lcb]
    return plan


# ---------------------------------------------------------------------------
# Binary-evaluation statistics
# ---------------------------------------------------------------------------

def confusion(labels, predictions):
    labels, predictions = np.asarray(labels), np.asarray(predictions)
    tp = int(((predictions == 1) & (labels == 1)).sum())
    tn = int(((predictions == 0) & (labels == 0)).sum())
    fp = int(((predictions == 1) & (labels == 0)).sum())
    fn = int(((predictions == 0) & (labels == 1)).sum())
    n = len(labels)
    return {"n": n, "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "accuracy": 100 * (tp + tn) / n if n else 0.0,
            "precision": 100 * tp / (tp + fp) if tp + fp else 0.0,
            "recall": 100 * tp / (tp + fn) if tp + fn else 0.0,
            "f1": 200 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
            "yes_rate": 100 * (tp + fp) / n if n else 0.0}


def mcnemar_exact(a_correct, b_correct):
    """Two-sided exact McNemar test on paired correctness (binomial on discordants)."""
    a, b = np.asarray(a_correct, bool), np.asarray(b_correct, bool)
    only_b, only_a = int((b & ~a).sum()), int((a & ~b).sum())
    n = only_a + only_b
    if n == 0:
        return {"b_only_correct": only_b, "a_only_correct": only_a, "p_value": 1.0}
    k = min(only_a, only_b)
    # P(X <= k) for X ~ Bin(n, 1/2), computed in log space for stability.
    log_terms = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) - n * math.log(2)
                 for i in range(k + 1)]
    m = max(log_terms)
    tail = math.exp(m) * sum(math.exp(t - m) for t in log_terms)
    return {"b_only_correct": only_b, "a_only_correct": only_a, "p_value": min(1.0, 2 * tail)}


def cluster_bootstrap_delta(image_ids, labels, pred_a, pred_b, *, draws, seed,
                            metrics=("accuracy", "f1", "recall", "precision")):
    """Percentile 95% CI of metric(b) - metric(a), resampling images."""
    image_ids = np.asarray(image_ids)
    labels, pred_a, pred_b = map(np.asarray, (labels, pred_a, pred_b))
    unique, index = np.unique(image_ids, return_inverse=True)
    # Per-image confusion counts for both methods: [I, 2 methods, 4 cells].
    cells = np.zeros((len(unique), 2, 4))
    for m, pred in enumerate((pred_a, pred_b)):
        for j, (p, y) in enumerate(((1, 1), (0, 0), (1, 0), (0, 1))):
            np.add.at(cells[:, m, j], index, ((pred == p) & (labels == y)).astype(float))
    rng = np.random.default_rng(seed)
    weights = np.stack([np.bincount(rng.integers(len(unique), size=len(unique)), minlength=len(unique))
                        for _ in range(draws)]).astype(float)
    totals = np.einsum("bi,imc->bmc", weights, cells)
    tp, tn, fp, fn = (totals[..., j] for j in range(4))
    with np.errstate(divide="ignore", invalid="ignore"):
        values = {"accuracy": 100 * (tp + tn) / (tp + tn + fp + fn),
                  "f1": 200 * tp / (2 * tp + fp + fn),
                  "recall": 100 * tp / (tp + fn),
                  "precision": 100 * tp / (tp + fp)}
    out = {}
    for metric in metrics:
        delta = values[metric][:, 1] - values[metric][:, 0]
        delta = delta[np.isfinite(delta)]
        lo, hi = np.quantile(delta, [0.025, 0.975])
        out[metric] = [float(lo), float(hi)]
    return out


def holm(p_values):
    """Holm-Bonferroni adjusted p-values, preserving input order."""
    order = sorted(range(len(p_values)), key=lambda i: p_values[i])
    adjusted, running = [0.0] * len(p_values), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p_values) - rank) * p_values[i]))
        adjusted[i] = running
    return adjusted


def threshold_for_rate(margins, rate):
    """Threshold tau such that mean(margin > tau) is as close as possible to rate."""
    values = np.sort(np.asarray(margins, dtype=float))[::-1]
    n = len(values)
    k = int(round(rate * n))
    if k <= 0:
        return float(values[0]) + 1e-6
    if k >= n:
        return float(values[-1]) - 1e-6
    return float((values[k - 1] + values[k]) / 2)


def best_threshold(margins, labels, *, min_recall=None):
    """Accuracy-maximizing threshold, optionally subject to a recall floor (in %)."""
    margins, labels = np.asarray(margins, float), np.asarray(labels, int)
    candidates = np.unique(np.concatenate([margins, [0.0]]))
    mids = np.concatenate([[candidates[0] - 1e-6], (candidates[:-1] + candidates[1:]) / 2,
                           [candidates[-1] + 1e-6]])
    best = None
    for tau in mids:
        stats = confusion(labels, (margins > tau).astype(int))
        if min_recall is not None and stats["recall"] < min_recall - 1e-9:
            continue
        key = (stats["accuracy"], -abs(tau))
        if best is None or key > best[0]:
            best = (key, float(tau))
    if best is None:
        raise ValueError("No threshold satisfies the recall floor")
    return best[1]
