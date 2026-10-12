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
from run_experiment import resolve_path, dataset_answer_score
from semantic_circuits.qwen_backend import Qwen25VL, QwenConfig
from semantic_circuits.runner import run_pathway_instance
from run_confirmatory_study import TeeLogger


def extract_predictors(r: dict) -> dict[str, float]:
    scc = r.get("scc")
    low_scc = 1.0 - float(scc) if scc is not None else 1.0
    cps = r.get("cps_weighted")
    low_cps = 1.0 - float(cps) if cps is not None else 1.0
    pbc = r.get("probe_behavior_consistency")
    probe_inconsistency = 1.0 - float(pbc) if pbc is not None else 1.0
    ext = r.get("external_intervention") or {}
    ext_nec = float(ext.get("necessity_js") or 0.0)
    ctrl_adj = max(0.0, float(ext.get("necessity_js_control_adjusted") or 0.0))
    dist = r.get("baseline_candidate_distribution") or {"other": 1.0}
    p_vals = np.asarray(list(dist.values()), dtype=float)
    p_vals = np.clip(p_vals, 1e-12, 1.0)
    p_vals = p_vals / p_vals.sum()
    low_conf = 1.0 - float(np.max(p_vals))
    ent = -np.sum(p_vals * np.log(p_vals))
    norm_ent = float(ent / max(np.log(len(p_vals)), 1e-12))
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
    eligible = [r for r in subset_rows if (r.get("vqa_consensus_score") or 0.0) >= 0.5 and r.get("heldout_failure") in (0, 1, False, True)]
    y_true = np.asarray([int(r["heldout_failure"]) for r in eligible])
    n_failures = int(np.sum(y_true))
    n_non_failures = len(y_true) - n_failures

    invariant = [r for r in eligible if r.get("probe_behavior_consistency") == 1.0]
    collapsed_invariant = [r for r in invariant if (r.get("cps_weighted") or 1.0) < 0.50]

    report = {
        "cohort": cohort_label,
        "total_instances": len(subset_rows),
        "baseline_correct": len(eligible),
        "baseline_accuracy": len(eligible) / max(len(subset_rows), 1),
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

    if len(np.unique(y_true)) < 2 or n_failures < 2:
        report["status"] = "Insufficient failure events for AUROC calculation"
        return report

    features = defaultdict(list)
    for r in eligible:
        preds = extract_predictors(r)
        for k, v in preds.items():
            features[k].append(v)

    rng = np.random.RandomState(seed)
    n_samples = len(y_true)
    results = {}

    for name, scores in features.items():
        y_score = np.asarray(scores)
        auroc = float(roc_auc_score(y_true, y_score))
        p_curve, r_curve, _ = precision_recall_curve(y_true, y_score)
        auprc = float(auc(r_curve, p_curve))
        brier = float(brier_score_loss(y_true, np.clip(y_score, 0, 1)))

        boot_aurocs, boot_auprcs = [], []
        for _ in range(1000):
            idx = rng.randint(0, n_samples, size=n_samples)
            if len(np.unique(y_true[idx])) < 2:
                continue
            boot_aurocs.append(float(roc_auc_score(y_true[idx], y_score[idx])))
            p_b, r_b, _ = precision_recall_curve(y_true[idx], y_score[idx])
            boot_auprcs.append(float(auc(r_b, p_b)))

        ci_auroc = [float(np.percentile(boot_aurocs, 2.5)), float(np.percentile(boot_aurocs, 97.5))] if boot_aurocs else [auroc, auroc]
        ci_auprc = [float(np.percentile(boot_auprcs, 2.5)), float(np.percentile(boot_auprcs, 97.5))] if boot_auprcs else [auprc, auprc]

        results[name] = {
            "auroc": auroc,
            "auroc_ci95": ci_auroc,
            "auprc": auprc,
            "auprc_ci95": ci_auprc,
            "brier": brier,
            "event_counts": {"failures": n_failures, "non_failures": n_non_failures},
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
            heldout_scores = [
                {**item, "vqa_consensus_score": dataset_answer_score(item["answer"], row.get("answers"), row.get("dataset"))}
                for item in result["heldout_variant_predictions"]
            ]
            score_threshold = float(config.get("failure_vqa_score_threshold", 0.5))
            future_failure = (
                int(any(item["vqa_consensus_score"] is None or item["vqa_consensus_score"] < score_threshold for item in heldout_scores))
                if baseline_vqa_score is not None and baseline_vqa_score >= score_threshold and heldout_scores
                else None
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
                "heldout_variant_predictions": heldout_scores,
                "heldout_failure": future_failure,
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

    all_rows = list(read_rows(manifest_path))
    print(f"Total manifest rows: {len(all_rows)}")

    train_rows = [r for r in all_rows if r.get("analysis_split") == "train"]
    val_rows = [r for r in all_rows if r.get("analysis_split") == "validation"]
    test_rows = [r for r in all_rows if r.get("analysis_split") == "test"]

    print(f"Partitions: Train={len(train_rows)} (200 VQA, 200 GQA), Validation={len(val_rows)} (100 VQA, 100 GQA), Locked Test={len(test_rows)} (200 VQA, 200 GQA)")

    # Initialize model
    print("\nInitializing Qwen2.5-VL-7B-Instruct (Frozen Target Architecture)...")
    model = Qwen25VL(QwenConfig(
        model_id=config["qwen_model"],
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
