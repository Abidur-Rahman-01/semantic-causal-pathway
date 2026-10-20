"""Fresh 1,000-Image Confirmatory Semantic Causal Pathway Study Orchestrator.

Protocol:
1. 1,000 images: 500 VQAv2 + 500 GQA (1 question per image).
2. 400 Train (200 VQA, 200 GQA), 200 Validation (100 VQA, 100 GQA), 400 Locked Test (200 VQA, 200 GQA).
3. Evaluates Train & Validation in Phase 1; freezes Phase 1 analysis.
4. Evaluates Locked 400 Test strictly once in Phase 2.
5. Reports VQAv2, GQA, and Combined separately.
6. Reports event counts (failures / non-failures) with every metric.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, brier_score_loss

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

os.environ["HF_HOME"] = str(ROOT / ".venv" / "hf_home")
os.environ["HUGGINGFACE_HUB_CACHE"] = str(ROOT / ".venv" / "hf_home" / "hub")

from analyze_experiment import read_rows
from benchmark_manifest import normalize_manifest_row, validate_rows
from run_experiment import resolve_path, dataset_answer_score
from semantic_circuits.qwen_backend import Qwen25VL, QwenConfig
from semantic_circuits.runner import run_pathway_instance
from run_confirmatory_study import TeeLogger


def extract_predictors(r: dict) -> dict[str, float]:
    scc = r.get("scc")
    low_scc = 1.0 - float(scc) if scc is not None else None
    cps = r.get("cps_weighted")
    low_cps = 1.0 - float(cps) if cps is not None else None
    pbc = r.get("probe_behavior_consistency")
    probe_inconsistency = 1.0 - float(pbc) if pbc is not None else None
    ext = r.get("external_intervention") or {}
    ext_nec = float(ext["necessity_js"]) if ext.get("necessity_js") is not None else None
    ctrl_adj = ext.get("necessity_js_control_adjusted")
    if ctrl_adj is None and ext_nec is not None and ext.get("control_necessity_mean_js") is not None:
        ctrl_adj = max(0.0, ext_nec - float(ext["control_necessity_mean_js"]))
    ctrl_adj = float(ctrl_adj) if ctrl_adj is not None else None
    dist = r.get("baseline_candidate_distribution") or {}
    p_vals = np.asarray(list(dist.values()), dtype=float) if dist else np.asarray([], dtype=float)
    if p_vals.size and np.isfinite(p_vals).all() and p_vals.sum() > 0:
        p_vals = np.clip(p_vals, 0, 1)
        p_vals = p_vals / p_vals.sum()
        low_conf = 1.0 - float(np.max(p_vals))
        ent = -np.sum(p_vals * np.log(np.clip(p_vals, 1e-12, 1.0)))
        norm_ent = float(ent / np.log(len(p_vals))) if len(p_vals) > 1 else 0.0
    else:
        low_conf, norm_ent = None, None
    return {
        "low_scc": low_scc,
        "low_cps": low_cps,
        "probe_behavioral_inconsistency": probe_inconsistency,
        "external_necessity": ext_nec,
        "control_adjusted_necessity": ctrl_adj,
        "low_confidence": low_conf,
        "normalized_entropy": norm_ent,
    }


def evaluate_cohort(subset_rows: list[dict], cohort_label: str, seed: int = 17) -> dict:
    baseline_correct = [r for r in subset_rows if r.get("vqa_consensus_score") is not None and r.get("vqa_consensus_score") >= 0.5]
    eligible = [r for r in baseline_correct if r.get("heldout_failure") in (0, 1, False, True)]
    y_true = np.asarray([int(r["heldout_failure"]) for r in eligible])
    n_failures = int(np.sum(y_true))
    n_non_failures = len(y_true) - n_failures

    invariant = [r for r in baseline_correct if r.get("probe_behavior_consistency") == 1.0]
    collapsed_invariant = [r for r in invariant if r.get("cps_weighted") is not None and float(r["cps_weighted"]) < 0.50]

    report = {
        "cohort": cohort_label,
        "total_instances": len(subset_rows),
        "baseline_correct": len(baseline_correct),
        "baseline_accuracy": len(baseline_correct) / max(len(subset_rows), 1),
        "event_counts": {
            "total_eligible": len(eligible),
            "failure_events": n_failures,
            "non_failure_events": n_non_failures,
        },
        "failure_prevalence": n_failures / max(len(eligible), 1),
        "behavioral_invariant_count": len(invariant),
        "invariant_collapsed_count": len(collapsed_invariant),
        "invariant_collapse_rate": len(collapsed_invariant) / max(len(invariant), 1),
    }

    from semantic_circuits.evaluation import grouped_bootstrap_binary_metrics
    results = {}
    for name in ("low_scc", "low_cps", "probe_behavioral_inconsistency", "external_necessity",
                 "control_adjusted_necessity", "low_confidence", "normalized_entropy"):
        valid = [r for r in eligible if extract_predictors(r).get(name) is not None]
        if not valid:
            results[name] = {"n": 0, "auroc": None, "auroc_ci95": None, "auprc": None,
                             "auprc_ci95": None, "brier": None, "valid_bootstrap_replicates": 0}
            continue
        metrics = grouped_bootstrap_binary_metrics(
            [int(r["heldout_failure"]) for r in valid],
            [extract_predictors(r)[name] for r in valid],
            [r.get("image_id", r.get("sample_id")) for r in valid], repeats=1000, seed=seed)
        results[name] = {
            "n": metrics["n"], "auroc": metrics["auroc"],
            "auroc_ci95": metrics["bootstrap"]["auroc"]["ci95"],
            "auprc": metrics["auprc"],
            "auprc_ci95": metrics["bootstrap"]["auprc"]["ci95"],
            "brier": metrics["brier"],
            "failure_prevalence": metrics["prevalence"],
            "valid_bootstrap_replicates": metrics["bootstrap"]["resamples_with_both_classes"],
        }
    report["predictors"] = results
    return report


def evaluate_split_comprehensive(all_rows: list[dict], split_name: str) -> dict:
    split_rows = [r for r in all_rows if r.get("analysis_split") == split_name]
    vqa_rows = [r for r in split_rows if "vqa" in r.get("dataset", "").lower()]
    gqa_rows = [r for r in split_rows if "gqa" in r.get("dataset", "").lower()]

    return {
        "split": split_name,
        "combined": evaluate_cohort(split_rows, f"{split_name}_combined"),
        "vqav2": evaluate_cohort(vqa_rows, f"{split_name}_vqav2"),
        "gqa": evaluate_cohort(gqa_rows, f"{split_name}_gqa"),
    }


def run_phase_inference(model, rows: list[dict], output_path: Path, manifest: Path, data_root: Path, config: dict) -> list[dict]:
    completed = set()
    existing_rows = []
    if output_path.exists():
        for r in read_rows(output_path):
            existing_rows.append(r)
            completed.add(str(r.get("sample_id")))

    print(f"Target phase rows: {len(rows)}. Already completed: {len(completed & {str(r.get('sample_id')) for r in rows})}")

    seed = int(config.get("seed", 17))
    random.seed(seed)
    np.random.seed(seed)

    count = 0
    with output_path.open("a", encoding="utf-8") as sink:
        for r_idx, row in enumerate(rows, 1):
            sample_id = str(row["sample_id"])
            if sample_id in completed:
                continue

            started = time.perf_counter()
            image_path = resolve_path(row["image"], manifest, data_root)
            mask_path = resolve_path(row["evidence_mask"], manifest, data_root)
            image = Image.open(image_path).convert("RGB")
            mask = np.asarray(Image.open(mask_path).convert("L")) > 0

            variants = []
            for v in row.get("variants", []):
                v_path = resolve_path(v["image"], manifest, data_root)
                loaded = {**v, "image": Image.open(v_path).convert("RGB")}
                if v.get("evidence_mask"):
                    loaded["evidence_mask"] = np.asarray(Image.open(resolve_path(v["evidence_mask"], manifest, data_root)).convert("L")) > 0
                variants.append(loaded)

            control_masks = []
            for c in row.get("control_masks", []):
                c_path = resolve_path(c["image"], manifest, data_root)
                control_masks.append(np.asarray(Image.open(c_path).convert("L")) > 0)

            answers = row.get("candidate_answers") or row.get("answers")

            result = run_pathway_instance(
                model, image, row["question"], mask, list(dict.fromkeys(answers)), variants,
                control_masks=control_masks,
                top_k=int(config.get("top_k", 20)),
                recovery_threshold=float(config.get("recovery_threshold", 0.8)),
                intervention_method=config.get("intervention_method", "blur"),
                heldout_failure_eval=True,
                mediator_search=config.get("mediator_search", "greedy"),
                exact_search_candidates=int(config.get("exact_search_candidates", 10)),
            )

            baseline_vqa_score = dataset_answer_score(result["baseline_answer"], row.get("answers"), row.get("dataset"))
            probe_pathways = [
                {**item, "dataset_score": dataset_answer_score(item["answer"], row.get("answers"), row.get("dataset"))}
                for item in result.get("variant_pathways", []) if item.get("transform") != "original"
            ]
            heldout_scores = [
                {**item, "vqa_consensus_score": dataset_answer_score(item["answer"], row.get("answers"), row.get("dataset"))}
                for item in result["heldout_variant_predictions"]
            ]
            score_threshold = float(config.get("failure_vqa_score_threshold", 0.5))
            heldout_scores_complete = bool(heldout_scores) and all(item.get("vqa_consensus_score") is not None for item in heldout_scores)
            future_failure = (
                int(any(item["vqa_consensus_score"] < score_threshold for item in heldout_scores))
                if baseline_vqa_score is not None and baseline_vqa_score >= score_threshold and heldout_scores_complete
                else None
            )
            heldout_answer_flip = (
                int(any(str(item.get("answer", "")).strip().casefold() != str(result["baseline_answer"]).strip().casefold()
                        for item in heldout_scores))
                if baseline_vqa_score is not None and baseline_vqa_score >= score_threshold and heldout_scores_complete else None
            )

            result.update({
                "sample_id": sample_id,
                "split": row.get("split"),
                "analysis_split": row.get("analysis_split"),
                "dataset": row.get("dataset"),
                "image_id": row.get("image_id"),
                "question_type": row.get("question_type"),
                "gold_answers": row.get("answers"),
                "vqa_consensus_score": baseline_vqa_score,
                "probe_variant_predictions": probe_pathways,
                "heldout_variant_predictions": heldout_scores,
                "heldout_failure": future_failure,
                "heldout_answer_flip": heldout_answer_flip,
                "failure_outcome_definition": "baseline_correct_and_any_approved_heldout_score_below_threshold",
                "failure_score_threshold": score_threshold,
                "scoring_function": f"dataset_answer_score:{row.get('dataset')}",
                "metric_undefined_reasons": {**(result.get("metric_undefined_reasons") or {}),
                    **({"heldout_failure": "one_or_more_heldout_scores_unavailable"} if not heldout_scores_complete else {})},
                "wall_seconds": time.perf_counter() - started,
            })

            sink.write(json.dumps(result, ensure_ascii=False) + "\n")
            sink.flush()
            existing_rows.append(result)
            count += 1
            print(f"  [{count}] Finished {sample_id} ({row.get('dataset')}, {row.get('analysis_split')}): Baseline={baseline_vqa_score}, HeldoutFail={future_failure}, SCC={result.get('scc')}, Wall={result['wall_seconds']:.1f}s", flush=True)

    return existing_rows


def main():
    print("=" * 80)
    print("Fresh 1,000-Image Confirmatory Semantic Causal Pathway Study Orchestrator")
    print("=" * 80)

    config_path = ROOT / "configs" / "pathway_study_1000.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    manifest_path = ROOT / config["manifest"]
    data_root = ROOT / config["data_root"]
    output_dir = ROOT / config["output_dir"] / "models" / "qwen2_5_vl_7b"
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "study_1000_experiment.log"
    sys.stdout = TeeLogger(log_file, sys.stdout)
    sys.stderr = TeeLogger(log_file, sys.stderr)
    results_path = output_dir / "semantic_pathway_results.jsonl"

    source_rows = list(read_rows(manifest_path))
    schema_v1 = any(row.get("schema_version") == "1.0" for row in source_rows)
    validation_errors, manifest_summary = validate_rows(
        source_rows, manifest_path, data_root,
        require_hashes=schema_v1, require_schema_v1=schema_v1)
    if validation_errors:
        raise ValueError("Benchmark manifest validation failed:\n" + "\n".join(validation_errors[:25]))
    (output_dir / "manifest_validation.json").write_text(json.dumps(manifest_summary, indent=2), encoding="utf-8")
    all_rows = [normalize_manifest_row(row) for row in source_rows]
    print(f"Total manifest rows: {len(all_rows)}")

    train_rows = [r for r in all_rows if r.get("analysis_split") == "train"]
    val_rows = [r for r in all_rows if r.get("analysis_split") == "validation"]
    test_rows = [r for r in all_rows if r.get("analysis_split") == "test"]

    print(f"Partitions: Train={len(train_rows)} (200 VQA, 200 GQA), Validation={len(val_rows)} (100 VQA, 100 GQA), Locked Test={len(test_rows)} (200 VQA, 200 GQA)")

    # Initialize model
    print("\nInitializing Qwen2.5-VL-7B-Instruct (Frozen Target Architecture)...")
    model = Qwen25VL(QwenConfig(
        model_id=config["qwen_model"],
        revision=config.get("qwen_revision"),
        device="auto",
        dtype="auto",
        max_new_tokens=int(config.get("max_new_tokens", 48)),
        max_image_pixels=config.get("max_image_pixels", 1003520),
    ))

    # Phase 1: Train & Validation
    print("\n" + "=" * 50)
    print("PHASE 1: Running Train & Validation Inference (600 images)")
    print("=" * 50)
    train_val_rows = train_rows + val_rows
    all_results = run_phase_inference(model, train_val_rows, results_path, manifest_path, data_root, config)

    # Analyze Phase 1
    print("\n" + "=" * 50)
    print("PHASE 1 ANALYSIS: Train & Validation Assessment (Freezing Analysis)")
    print("=" * 50)
    train_metrics = evaluate_split_comprehensive(all_results, "train")
    val_metrics = evaluate_split_comprehensive(all_results, "validation")
    phase1_report = {"train": train_metrics, "validation": val_metrics}

    phase1_path = output_dir / "phase1_train_val_report.json"
    phase1_path.write_text(json.dumps(phase1_report, indent=2), encoding="utf-8")
    print(f"Phase 1 report saved: {phase1_path}")
    print(json.dumps(phase1_report, indent=2))

    # Phase 2: Locked 400 Test Split (Strictly Evaluated Once)
    print("\n" + "=" * 50)
    print("PHASE 2: Evaluating Locked 400 Test Partition (Strictly Once, Fresh Images)")
    print("=" * 50)
    all_results = run_phase_inference(model, test_rows, results_path, manifest_path, data_root, config)

    print("\n" + "=" * 50)
    print("FINAL CONFIRMATORY EVALUATION ON LOCKED TEST SET (N = 400)")
    print("=" * 50)
    test_metrics = evaluate_split_comprehensive(all_results, "test")
    final_report = {
        "study_stage": "confirmatory_study_1000",
        "date": "2026-10-10",
        "model": "Qwen/Qwen2.5-VL-7B-Instruct",
        "train": train_metrics,
        "validation": val_metrics,
        "test": test_metrics,
    }

    final_path = output_dir / "study_1000_final_report.json"
    final_path.write_text(json.dumps(final_report, indent=2), encoding="utf-8")
    print(f"1,000-Image Confirmatory Report saved: {final_path}")
    print(json.dumps(final_report, indent=2))

    # Also save a copy to outputs/
    public_final_path = ROOT / "outputs" / "study_1000_final_report.json"
    public_final_path.write_text(json.dumps(final_report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
