"""Greedy and exact small-set mediator recovery."""
from __future__ import annotations
from itertools import combinations


def recover_greedy(candidates: list[dict], factual_score: float, counterfactual_score: float,
                   patch_score, recovery_threshold: float = 0.8) -> dict:
    """Select candidate layer/head mediators by restored target-answer log probability.

    patch_score receives a dict {layer_module_name: {activation, heads}} and returns
    the patched target-answer log probability. The search performs repeated forwards.
    """
    gap = factual_score - counterfactual_score
    if gap <= 0:
        return {"mediators": [], "recovery": 0.0, "reason": "counterfactual_did_not_reduce_target_score", "forward_evaluations": 0}
    selected: list[dict] = []
    unused = list(candidates)
    evaluations = 0

    def make_patch(items):
        patch = {}
        for item in items:
            name, head = item["layer"], int(item["head"])
            if name not in patch:
                patch[name] = {"activation": item["activation"], "heads": []}
            patch[name]["heads"].append(head)
        return patch

    current_score = counterfactual_score
    while unused and (current_score-counterfactual_score)/gap < recovery_threshold:
        trials = []
        for candidate in unused:
            trials.append((float(patch_score(make_patch(selected + [candidate]))), candidate))
            evaluations += 1
        best_score, best = max(trials, key=lambda pair: pair[0])
        if best_score <= current_score:
            break
        selected.append(best)
        unused.remove(best)
        current_score = best_score

    # Backward prune to report a compact approximate mediator set.
    changed = True
    while changed and len(selected) > 1:
        changed = False
        for item in list(selected):
            trial = [x for x in selected if x is not item]
            score = float(patch_score(make_patch(trial)))
            evaluations += 1
            if (score-counterfactual_score)/gap >= recovery_threshold:
                selected, current_score, changed = trial, score, True
                break
    return {"mediators": [{"layer": x["layer"], "head": int(x["head"]), "attribution_norm": float(x.get("attribution_norm", 0))} for x in selected],
            "recovered_target_logprob": current_score,
            "recovery_fraction": max(0.0, (current_score-counterfactual_score)/gap),
            "recovery_threshold": recovery_threshold,
            "forward_evaluations": evaluations,
            "optimality": "greedy_forward_selection_with_backward_pruning"}


def exact_small_search(candidates: list[dict], factual_score: float, counterfactual_score: float,
                       patch_score, recovery_threshold: float = 0.8, max_candidates: int = 12) -> dict:
    """Exact cardinality-ordered search for <=12 candidate units (expensive)."""
    if len(candidates) > max_candidates:
        raise ValueError(f"Exact search limited to {max_candidates} units")
    gap = factual_score-counterfactual_score
    if gap <= 0:
        return {"mediators": [], "recovery_fraction": 0.0, "forward_evaluations": 0}
    evaluations = 0
    for size in range(len(candidates)+1):
        for subset in combinations(candidates, size):
            patch = {}
            for item in subset:
                name, head = item["layer"], int(item["head"])
                patch.setdefault(name, {"activation": item["activation"], "heads": []})["heads"].append(head)
            score = counterfactual_score if not subset else float(patch_score(patch))
            evaluations += bool(subset)
            recovery = (score-counterfactual_score)/gap
            if recovery >= recovery_threshold:
                return {"mediators": [{"layer": x["layer"], "head": int(x["head"])} for x in subset],
                        "recovery_fraction": recovery, "forward_evaluations": evaluations,
                        "optimality": "exact_minimum_cardinality_within_candidate_set"}
    return {"mediators": [], "recovery_fraction": 0.0, "forward_evaluations": evaluations,
            "optimality": "no_sufficient_subset_found"}
