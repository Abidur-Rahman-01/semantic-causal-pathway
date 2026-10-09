"""CPS, CECA, SCC, and uncertainty-aware metric primitives."""
from __future__ import annotations

import math
from itertools import combinations


def _normalize(values):
    values = {k: max(0.0, float(v)) for k, v in values.items()}
    total = sum(values.values())
    return {k: v / total for k, v in values.items()} if total else {}


def js_divergence(p: dict, q: dict, eps: float = 1e-12) -> float:
    """Jensen-Shannon divergence (natural logs), returned in [0, ln(2)]."""
    keys = set(p) | set(q)
    if not keys:
        return 0.0
    p = _normalize({k: p.get(k, 0.0) for k in keys})
    q = _normalize({k: q.get(k, 0.0) for k in keys})
    if not p or not q:
        return math.log(2)
    m = {k: (p.get(k, 0.0) + q.get(k, 0.0)) / 2 for k in keys}
    def kl(a, b):
        return sum(x * math.log((x + eps) / (b.get(k, 0.0) + eps)) for k, x in a.items() if x > 0)
    return max(0.0, (kl(p, m) + kl(q, m)) / 2)


def cps_hard(mediator_sets: list[set]) -> float | None:
    """Mean pairwise Jaccard similarity of recovered mediator sets."""
    if len(mediator_sets) < 2:
        return None
    scores = []
    for left, right in combinations(mediator_sets, 2):
        union = left | right
        # Two empty recovered sets mean no identified circuit, not perfect stability.
        scores.append(len(left & right) / len(union) if union else 0.0)
    return sum(scores) / len(scores)


def cps_weighted(mediator_distributions: list[dict]) -> float | None:
    """Mean pairwise 1 - normalized-JS similarity for mediator effect distributions."""
    if len(mediator_distributions) < 2:
        return None
    ceiling = math.log(2)
    pair_scores = []
    for a, b in combinations(mediator_distributions, 2):
        if not a or not b:
            pair_scores.append(0.0)
        else:
            pair_scores.append(1 - min(1.0, js_divergence(a, b) / ceiling))
    return sum(pair_scores) / len(pair_scores)


def ceca_scalar(external_effect: float, mediated_effect: float, eps: float = 1e-12) -> float:
    """Scale agreement of nonnegative external and mediated effect magnitudes."""
    a, b = max(0.0, float(external_effect)), max(0.0, float(mediated_effect))
    if max(a, b) <= eps:
        return 0.0
    return min(a, b) / (max(a, b) + eps)


def ceca_distribution(external_effects: dict, mediated_effects: dict) -> float:
    """Normalized-JS agreement; effect keys must denote the same outcome bins."""
    if not external_effects or not mediated_effects:
        return 0.0
    return 1 - min(1.0, js_divergence(external_effects, mediated_effects) / math.log(2))


def positive_distribution_shift(reference: dict, counterfactual: dict) -> dict:
    """Normalize positive per-answer probability changes over one fixed answer set."""
    return _normalize({key: max(0.0, float(reference.get(key, 0))-float(counterfactual.get(key, 0)))
                       for key in set(reference) | set(counterfactual)})


def ceca_from_output_distributions(full: dict, deleted: dict, patched: dict) -> float | None:
    """Align external necessity and internal restoration on the same answer bins."""
    external = positive_distribution_shift(full, deleted)
    mediated = positive_distribution_shift(patched, deleted)
    if not external or not mediated:
        return None
    return ceca_distribution(external, mediated)


def scc_score(ceca: float, cps: float) -> float:
    return max(0.0, min(1.0, float(ceca))) * max(0.0, min(1.0, float(cps)))
