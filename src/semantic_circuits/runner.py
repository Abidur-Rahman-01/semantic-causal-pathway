"""One-sample semantic causal pathway experiment orchestration."""
from __future__ import annotations

from .interventions import external_effects, output_distribution, replace_mask
from .mediator_search import recover_greedy
from .metrics import ceca_scalar, ceca_from_output_distributions, cps_hard, cps_weighted, scc_score
from .qwen_backend import rank_attention_heads
from .evidence import parse_question_requirements


def run_pathway_instance(model, image, question: str, evidence_mask, candidate_answers: list[str],
                         variants: list[dict], control_masks: list | None = None,
                         top_k: int = 20, recovery_threshold: float = 0.8,
                         intervention_method: str = "blur", heldout_failure_eval: bool = False) -> dict:
    """Run baseline, external effect, per-variant mediator recovery, CPS/CECA/SCC.

    Each variant must have `image`, `transform`, and all four validity fields set True.
    Same evidence mask coordinates/resolution are assumed; re-ground after any geometry edit.
    """
    from .transforms import validity_gate
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        pass
    accepted, rejected = [], []
    for variant in variants:
        valid, reasons = validity_gate(variant)
        if variant["image"].size != image.size:
            valid, reasons = False, [*reasons, "image_geometry_changed_requires_regrounding"]
        (accepted if valid else rejected).append((variant, reasons))
    tagged = any(v.get("analysis_role") for v, _ in accepted)
    if heldout_failure_eval and not tagged:
        raise ValueError("Held-out evaluation requires human-assigned probe/heldout roles on approved variants.")
    probe = [(v, reasons) for v, reasons in accepted if not tagged or v.get("analysis_role") == "probe"]
    heldout = [(v, reasons) for v, reasons in accepted if tagged and v.get("analysis_role") == "heldout"]
    if not probe:
        raise ValueError("CPS needs at least one approved probe variant plus the original image.")
    if heldout_failure_eval and not heldout:
        raise ValueError("Held-out evaluation needs at least one approved heldout variant.")
    mask_shape = getattr(evidence_mask, "shape", None)
    if mask_shape is None and hasattr(evidence_mask, "size"):
        mask_shape = (evidence_mask.size[1], evidence_mask.size[0])
    if mask_shape != (image.height, image.width):
        raise ValueError("Evidence mask dimensions must match the original image.")
    start_calls = model.forward_calls
    start_seconds = model.forward_seconds
    pred = model.answer(image, question)
    if pred not in candidate_answers:
        candidate_answers = [pred, *candidate_answers]
    base_scores = model.answer_logprobs(image, question, candidate_answers)
    external = external_effects(image, question, evidence_mask, candidate_answers,
                                model.answer_logprobs, intervention_method, control_masks)
    run_variants = [{"transform": "original", "image": image}, *[v for v, _ in probe]]
    mediator_sets, mediator_weights, variant_rows = [], [], []
    for record in run_variants:
        current = record["image"].convert("RGB")
        current_mask = record.get("evidence_mask", evidence_mask)
        mask_shape = getattr(current_mask, "shape", None)
        if mask_shape is None and hasattr(current_mask, "size"):
            mask_shape = (current_mask.size[1], current_mask.size[0])
        if mask_shape != (current.height, current.width):
            raise ValueError(f"Evidence mask dimensions must match {record['transform']} image.")
        variant_answer = model.answer(current, question)
        cf_image = replace_mask(current, current_mask, intervention_method)
        factual_logs = model.answer_logprobs(current, question, candidate_answers)
        cf_logs = model.answer_logprobs(cf_image, question, candidate_answers)
        factual_score, cf_score = factual_logs[pred], cf_logs[pred]
        fact_acts = model.capture_answer_activations(current, question, pred)
        cf_acts = model.capture_answer_activations(cf_image, question, pred)
        heads = model.num_attention_heads
        ranked = rank_attention_heads(fact_acts, cf_acts, heads, top_k=top_k)
        candidates = [{**row, "activation": fact_acts[row["layer"]]} for row in ranked]
        patch_score = lambda patch: model.patched_answer_logprob(cf_image, question, pred, patch)
        recovered = recover_greedy(candidates, factual_score, cf_score, patch_score,
                                   recovery_threshold=recovery_threshold)
        mediated_effect = max(0.0, recovered.get("recovered_target_logprob", cf_score) - cf_score)
        external_effect = max(0.0, factual_score - cf_score)
        ceca = ceca_scalar(external_effect, mediated_effect)
        selected_patch = {}
        for selected in recovered["mediators"]:
            candidate = next(x for x in candidates if x["layer"] == selected["layer"] and x["head"] == selected["head"])
            entry = selected_patch.setdefault(selected["layer"], {"activation": candidate["activation"], "heads": []})
            entry["heads"].append(selected["head"])
        patched_logs = model.patched_answer_logprobs(cf_image, question, candidate_answers, selected_patch) if selected_patch else cf_logs
        p_full, p_deleted, p_patched = map(output_distribution, (factual_logs, cf_logs, patched_logs))
        ceca_dist = ceca_from_output_distributions(p_full, p_deleted, p_patched)
        # Spatial controls are measured on the original frame; translated variants have shifted coordinates.
        external_variant = external_effects(current, question, current_mask, candidate_answers,
                                             model.answer_logprobs, method=intervention_method,
                                             control_masks=None)
        effects = {}
        for item in recovered["mediators"]:
            key = f"{item['layer']}#head{item['head']}"
            candidate_row = next(x for x in candidates if x["layer"] == item["layer"] and x["head"] == item["head"])
            single_patch = {item["layer"]: {"activation": candidate_row["activation"], "heads": [item["head"]]}}
            single_score = model.patched_answer_logprob(cf_image, question, pred, single_patch)
            # Weighted CPS is based on isolated mediated log-probability restoration,
            # not the attribution ranking used only to prune the candidate search.
            effects[key] = max(0.0, single_score - cf_score)
        mediator_sets.append(set(effects))
        mediator_weights.append(effects)
        variant_rows.append({"transform": record["transform"], "answer": variant_answer,
            "answer_matches_baseline": variant_answer.strip().casefold() == pred.strip().casefold(),
            "factual_target_logprob": factual_score,
            "counterfactual_target_logprob": cf_score, "mediated_target_logprob": recovered.get("recovered_target_logprob"),
            "external_target_effect": external_effect, "mediated_effect": mediated_effect,
            "ceca_scalar": ceca, "ceca_distribution": ceca_dist,
            "external_intervention": external_variant,
            "mediator_recovery": recovered,
            "top_attribution_heads": [{k:v for k,v in row.items() if k != "activation"} for row in ranked]})
    hard = cps_hard(mediator_sets)
    weighted = cps_weighted(mediator_weights)
    ceca_values = [row["ceca_distribution"] for row in variant_rows if row["ceca_distribution"] is not None]
    ceca_mean = sum(ceca_values) / len(ceca_values) if ceca_values else None
    ceca_scalar_mean = sum(row["ceca_scalar"] for row in variant_rows) / len(variant_rows)
    scc = scc_score(ceca_mean, weighted if weighted is not None else 0.0) if ceca_mean is not None else None
    probe_predictions = [row["answer"] for row in variant_rows if row["transform"] != "original"]
    probe_consistency = (sum(answer.strip().casefold() == pred.strip().casefold()
                             for answer in probe_predictions) / len(probe_predictions)
                         if probe_predictions else None)
    heldout_predictions = []
    for record, _ in heldout:
        heldout_predictions.append({"transform": record["transform"],
                                    "answer": model.answer(record["image"].convert("RGB"), question)})
    cost = {"model_forward_calls": model.forward_calls-start_calls,
            "model_forward_seconds": model.forward_seconds-start_seconds,
            "batch_size": 1}
    try:
        import torch
        if torch.cuda.is_available():
            cost["cuda_peak_memory_bytes"] = int(torch.cuda.max_memory_allocated())
    except ImportError:
        pass
    return {"model_id": model.config.model_id, "question": question,
        "requirements": parse_question_requirements(question), "baseline_answer": pred,
        "baseline_candidate_logprobs": base_scores, "baseline_candidate_distribution": output_distribution(base_scores),
        "external_intervention": external, "accepted_variant_count": len(accepted),
        "rejected_variants": [{"transform": v.get("transform"), "failed_gate_fields": reasons} for v,reasons in rejected],
        "variant_pathways": variant_rows, "cps_hard": hard, "cps_weighted": weighted,
        "probe_behavior_consistency": probe_consistency,
        "heldout_variant_predictions": heldout_predictions,
        "ceca_distribution_mean": ceca_mean, "ceca_scalar_mean_diagnostic": ceca_scalar_mean, "scc": scc,
        "cost": cost,
        "limitations": ["approximate mediator set", "attention-head input interventions only",
                        "CECA scalar is diagnostic; use a preregistered effect-distribution design for primary inference"]}
