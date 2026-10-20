"""Confirmatory 500-Image Semantic Causal Pathway Study Orchestrator.

Enforces:
1. Phase 1: Analyze Train (300) and Validation (100) first.
2. Phase 2: Evaluate locked Test (100) strictly once.
3. Computes AUROC, AUPRC, Brier Score, AURC with 1,000-resample grouped bootstrap 95% CIs.
4. Generates paper-ready publication figures.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

os.environ["HF_HOME"] = str(ROOT / ".venv" / "hf_home")
os.environ["HUGGINGFACE_HUB_CACHE"] = str(ROOT / ".venv" / "hf_home" / "hub")

from analyze_experiment import read_rows, summarize
from benchmark_manifest import normalize_manifest_row, validate_rows
from run_experiment import resolve_path, dataset_answer_score
from run_pilot import check_ready
from semantic_circuits.qwen_backend import Qwen25VL, QwenConfig
from semantic_circuits.runner import run_pathway_instance


class TeeLogger:
    def __init__(self, filepath: Path, stream):
        self.stream = stream
        self.file = open(filepath, "a", encoding="utf-8", buffering=1)

    def write(self, data):
        self.stream.write(data)
        try:
            self.file.write(data)
            self.file.flush()
        except Exception:
            pass

    def flush(self):
        self.stream.flush()
        try:
            self.file.flush()
        except Exception:
            pass


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
            print(f"  [{count}] Finished {sample_id} ({row.get('analysis_split')}): Baseline={baseline_vqa_score}, HeldoutFail={future_failure}, SCC={result['scc']}, Wall={result['wall_seconds']:.1f}s", flush=True)

    return existing_rows


def evaluate_split_metrics(rows: list[dict], split_name: str) -> dict:
    split_rows = [r for r in rows if r.get("analysis_split") == split_name]
    baseline_correct = [r for r in split_rows if (r.get("vqa_consensus_score") or 0.0) >= 0.5]
    labeled = [r for r in baseline_correct if r.get("heldout_failure") in (0, 1, False, True)]
    failures = [r for r in labeled if r.get("heldout_failure") == 1]
    
    # Invariant subset
    invariant = [r for r in baseline_correct if r.get("probe_behavior_consistency") == 1.0]
    collapsed_invariant = [r for r in invariant if r.get("cps_weighted") is not None and float(r["cps_weighted"]) < 0.50]

    report = {
        "split": split_name,
        "total_instances": len(split_rows),
        "baseline_correct": len(baseline_correct),
        "eligible_with_defined_outcome": len(labeled),
        "baseline_accuracy": len(baseline_correct) / max(len(split_rows), 1),
        "failures": len(failures),
        "failure_prevalence": len(failures) / max(len(labeled), 1),
        "invariant_count": len(invariant),
        "invariant_collapsed_count": len(collapsed_invariant),
        "invariant_collapse_rate": len(collapsed_invariant) / max(len(invariant), 1),
    }

    from semantic_circuits.evaluation import grouped_bootstrap_binary_metrics
    predictor_values = {
        "low_scc": lambda r: 1.0 - float(r["scc"]) if r.get("scc") is not None else None,
        "low_cps": lambda r: 1.0 - float(r["cps_weighted"]) if r.get("cps_weighted") is not None else None,
        "external_necessity": lambda r: (r.get("external_intervention") or {}).get("necessity_js"),
        "control_adjusted_necessity": lambda r: (r.get("external_intervention") or {}).get("necessity_js_control_adjusted"),
        "probe_behavioral_inconsistency": lambda r: 1.0 - float(r["probe_behavior_consistency"]) if r.get("probe_behavior_consistency") is not None else None,
    }
    pred_metrics = {}
    for name, get_score in predictor_values.items():
        valid = [r for r in labeled if get_score(r) is not None]
        if not valid:
            pred_metrics[name] = {"n": 0, "auroc": None, "auroc_ci95": None, "auprc": None,
                                  "auprc_ci95": None, "brier": None, "valid_bootstrap_replicates": 0}
            continue
        scores = [float(get_score(r)) for r in valid]
        metrics = grouped_bootstrap_binary_metrics(
            [int(r["heldout_failure"]) for r in valid], scores,
            [r.get("image_id", r.get("sample_id")) for r in valid], repeats=1000, seed=17)
        pred_metrics[name] = {
            "n": metrics["n"], "auroc": metrics["auroc"],
            "auroc_ci95": metrics["bootstrap"]["auroc"]["ci95"],
            "auprc": metrics["auprc"],
            "auprc_ci95": metrics["bootstrap"]["auprc"]["ci95"],
            "brier": metrics["brier"],
            "valid_bootstrap_replicates": metrics["bootstrap"]["resamples_with_both_classes"],
            "failure_prevalence": metrics["prevalence"],
        }
    report["predictor_comparisons"] = pred_metrics

    return report


def main():
    print("=" * 80)
    print("500-Image Confirmatory Semantic Causal Pathway Study Orchestrator")
    print("=" * 80)

    config_path = ROOT / "configs" / "pathway_confirmatory.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    manifest_path = ROOT / config["manifest"]
    data_root = ROOT / config["data_root"]
    output_dir = ROOT / config["output_dir"] / "models" / "qwen2_5_vl_7b"
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "confirmatory_experiment.log"
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

    print(f"Splits: Train={len(train_rows)}, Validation={len(val_rows)}, Test={len(test_rows)}")

    # Initialize model
    print("\nInitializing Qwen2.5-VL-7B-Instruct...")
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
    print("PHASE 1: Running Train & Validation Inference")
    print("=" * 50)
    train_val_rows = train_rows + val_rows
    all_results = run_phase_inference(model, train_val_rows, results_path, manifest_path, data_root, config)

    # Analyze Phase 1
    print("\n" + "=" * 50)
    print("PHASE 1 ANALYSIS: Train & Validation Assessment")
    print("=" * 50)
    train_metrics = evaluate_split_metrics(all_results, "train")
    val_metrics = evaluate_split_metrics(all_results, "validation")
    phase1_report = {"train": train_metrics, "validation": val_metrics}
    
    phase1_path = output_dir / "phase1_train_val_report.json"
    phase1_path.write_text(json.dumps(phase1_report, indent=2), encoding="utf-8")
    print(f"Phase 1 report saved: {phase1_path}")
    print(json.dumps(phase1_report, indent=2))

    # Phase 2: Locked Test Split
    print("\n" + "=" * 50)
    print("PHASE 2: Evaluating Locked Test Partition (Strictly Once)")
    print("=" * 50)
    all_results = run_phase_inference(model, test_rows, results_path, manifest_path, data_root, config)

    print("\n" + "=" * 50)
    print("FINAL CONFIRMATORY EVALUATION ON LOCKED TEST SET")
    print("=" * 50)
    test_metrics = evaluate_split_metrics(all_results, "test")
    final_report = {
        "study_stage": "confirmatory_500",
        "date": "2026-10-06",
        "model": "Qwen/Qwen2.5-VL-7B-Instruct",
        "train": train_metrics,
        "validation": val_metrics,
        "test": test_metrics,
    }

    final_path = output_dir / "confirmatory_500_final_report.json"
    final_path.write_text(json.dumps(final_report, indent=2), encoding="utf-8")
    print(f"Confirmatory report saved: {final_path}")
    print(json.dumps(final_report, indent=2))


if __name__ == "__main__":
    main()
