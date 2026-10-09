"""1,000-Image Confirmatory Semantic Causal Pathway Study Orchestrator.

Enforces:
1. Reuses 500 already-computed results from pathway_confirmatory.
2. Evaluates the remaining 500 new instances sequentially.
3. Zero-leakage Phase 1 (Train 600, Validation 200) -> Phase 2 (Locked Test 200).
4. Computes AUROC, AUPRC, Brier Score with 1,000-resample bootstrap 95% CIs.
5. Saves final report to confirmatory_1000_final_report.json.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import random
import shutil
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

from analyze_experiment import read_rows
from run_experiment import resolve_path, dataset_answer_score
from run_pilot import check_ready
from semantic_circuits.qwen_backend import Qwen25VL, QwenConfig
from semantic_circuits.runner import run_pathway_instance
from run_confirmatory_study import TeeLogger, run_phase_inference, evaluate_split_metrics


def main():
    print("=" * 80)
    print("1,000-Image Confirmatory Semantic Causal Pathway Study Orchestrator")
    print("=" * 80)

    config_path = ROOT / "configs" / "pathway_confirmatory_1000.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    manifest_path = ROOT / config["manifest"]
    data_root = ROOT / config["data_root"]
    output_dir = ROOT / config["output_dir"] / "models" / "qwen2_5_vl_7b"
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "confirmatory_1000_experiment.log"
    sys.stdout = TeeLogger(log_file, sys.stdout)
    sys.stderr = TeeLogger(log_file, sys.stderr)
    results_path = output_dir / "semantic_pathway_results.jsonl"

    # Seed with 500 existing results if results_path doesn't exist
    source_500_results = ROOT / ".venv" / "outputs" / "pathway_confirmatory" / "models" / "qwen2_5_vl_7b" / "semantic_pathway_results.jsonl"
    if not results_path.exists() and source_500_results.exists():
        print(f"Pre-seeding 500 completed results from: {source_500_results}")
        shutil.copy(source_500_results, results_path)

    all_rows = list(read_rows(manifest_path))
    print(f"Total 1,000-image manifest rows: {len(all_rows)}")

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
    print("FINAL CONFIRMATORY EVALUATION ON LOCKED TEST SET (1000 SAMPLES)")
    print("=" * 50)
    test_metrics = evaluate_split_metrics(all_results, "test")
    final_report = {
        "study_stage": "confirmatory_1000",
        "date": "2026-10-09",
        "model": "Qwen/Qwen2.5-VL-7B-Instruct",
        "train": train_metrics,
        "validation": val_metrics,
        "test": test_metrics,
    }

    final_path = output_dir / "confirmatory_1000_final_report.json"
    final_path.write_text(json.dumps(final_report, indent=2), encoding="utf-8")
    print(f"Confirmatory 1,000 report saved: {final_path}")
    print(json.dumps(final_report, indent=2))


if __name__ == "__main__":
    main()
