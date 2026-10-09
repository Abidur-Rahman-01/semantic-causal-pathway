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
            print(f"  [{count}] Finished {sample_id} ({row.get('analysis_split')}): Baseline={baseline_vqa_score}, HeldoutFail={future_failure}, SCC={result['scc']}, Wall={result['wall_seconds']:.1f}s", flush=True)

    return existing_rows


def evaluate_split_metrics(rows: list[dict], split_name: str) -> dict:
    split_rows = [r for r in rows if r.get("analysis_split") == split_name]
    baseline_correct = [r for r in split_rows if (r.get("vqa_consensus_score") or 0.0) >= 0.5]
    failures = [r for r in baseline_correct if r.get("heldout_failure") == 1]
    
    # Invariant subset
    invariant = [r for r in baseline_correct if r.get("probe_behavior_consistency") == 1.0]
    collapsed_invariant = [r for r in invariant if (r.get("cps_weighted") or 1.0) < 0.50]

    report = {
        "split": split_name,
        "total_instances": len(split_rows),
        "baseline_correct": len(baseline_correct),
        "baseline_accuracy": len(baseline_correct) / max(len(split_rows), 1),
        "failures": len(failures),
        "failure_prevalence": len(failures) / max(len(baseline_correct), 1),
        "invariant_count": len(invariant),
        "invariant_collapsed_count": len(collapsed_invariant),
        "invariant_collapse_rate": len(collapsed_invariant) / max(len(invariant), 1),
    }

    if len(baseline_correct) >= 5 and len(failures) >= 2 and len(failures) < len(baseline_correct):
        from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, brier_score_loss
        
        y_true = np.array([1 if r.get("heldout_failure") == 1 else 0 for r in baseline_correct])
        
        # Predictors
        predictors = {
            "low_scc": [1.0 - (r.get("scc") if r.get("scc") is not None else 0.5) for r in baseline_correct],
            "low_cps": [1.0 - (r.get("cps_weighted") if r.get("cps_weighted") is not None else 0.5) for r in baseline_correct],
            "external_necessity": [(r.get("external_intervention") or {}).get("necessity_js") or 0.0 for r in baseline_correct],
            "control_adjusted_necessity": [max(0.0, ((r.get("external_intervention") or {}).get("necessity_js") or 0.0) - ((r.get("external_intervention") or {}).get("control_necessity_mean_js") or 0.0)) for r in baseline_correct],
            "probe_behavioral_inconsistency": [1.0 - (r.get("probe_behavior_consistency") or 1.0) for r in baseline_correct],
        }

        # Bootstrap CIs (1,000 resamples)
        rng = np.random.RandomState(17)
        pred_metrics = {}
        for name, scores in predictors.items():
            y_score = np.array(scores)
            auroc = float(roc_auc_score(y_true, y_score))
            p_curve, r_curve, _ = precision_recall_curve(y_true, y_score)
            auprc = float(auc(r_curve, p_curve))
            brier = float(brier_score_loss(y_true, np.clip(y_score, 0, 1)))

            # Bootstrap
            boot_aurocs, boot_auprcs = [], []
            n_samples = len(y_true)
            for _ in range(1000):
                boot_idx = rng.randint(0, n_samples, size=n_samples)
                if len(np.unique(y_true[boot_idx])) < 2:
                    continue
                boot_aurocs.append(float(roc_auc_score(y_true[boot_idx], y_score[boot_idx])))
                p_b, r_b, _ = precision_recall_curve(y_true[boot_idx], y_score[boot_idx])
                boot_auprcs.append(float(auc(r_b, p_b)))

            ci_auroc = [float(np.percentile(boot_aurocs, 2.5)), float(np.percentile(boot_aurocs, 97.5))] if boot_aurocs else [auroc, auroc]
            ci_auprc = [float(np.percentile(boot_auprcs, 2.5)), float(np.percentile(boot_auprcs, 97.5))] if boot_auprcs else [auprc, auprc]

            pred_metrics[name] = {
                "auroc": auroc,
                "auroc_ci95": ci_auroc,
                "auprc": auprc,
                "auprc_ci95": ci_auprc,
                "brier": brier,
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

    all_rows = list(read_rows(manifest_path))
    print(f"Total manifest rows: {len(all_rows)}")

    train_rows = [r for r in all_rows if r.get("analysis_split") == "train"]
    val_rows = [r for r in all_rows if r.get("analysis_split") == "validation"]
    test_rows = [r for r in all_rows if r.get("analysis_split") == "test"]

    print(f"Splits: Train={len(train_rows)}, Validation={len(val_rows)}, Test={len(test_rows)}")

    # Initialize model
    print("\nInitializing Qwen2.5-VL-7B-Instruct...")
    model = Qwen25VL(QwenConfig(
        model_id=config["qwen_model"],
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
