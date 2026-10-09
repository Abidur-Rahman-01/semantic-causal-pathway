"""Orchestrate training on 30K samples with 10 epochs and evaluation on >=30K test samples.

Follows the Semantic Causal Pathway Consistency MLLM Architecture specification.
Optimized for NVIDIA RTX 3090 (24GB VRAM):
- bfloat16 mixed precision
- Gradient checkpointing
- Batch size 2 with grad accumulation 4 (effective batch 8)
- Cosine annealing learning rate schedule
- Evaluation on >= 30,000 held-out test samples with official consensus scoring
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


def run_cmd(cmd: list[str], desc: str) -> None:
    print(f"\n{'='*75}\n[RUNNING] {desc}\nCOMMAND: {' '.join(str(c) for c in cmd)}\n{'='*75}", flush=True)
    t0 = time.time()
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.run(cmd, cwd=str(ROOT), env=env)
    elapsed = time.time() - t0
    if proc.returncode != 0:
        raise RuntimeError(f"Step failed ({desc}) with returncode {proc.returncode} in {elapsed:.1f}s")
    print(f"[COMPLETED] {desc} in {elapsed:.1f}s", flush=True)


def prepare_30k_splits(train_full: Path, test_full: Path, out_dir: Path,
                       train_samples: int = 30000, test_samples: int = 30000) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    train_30k_path = out_dir / f"train_{train_samples//1000}k.jsonl"
    test_30k_path = out_dir / f"locked_test_{test_samples//1000}k.jsonl"

    if not train_30k_path.exists():
        print(f"Preparing {train_samples:,} training samples from {train_full}...")
        rows = [json.loads(line) for line in train_full.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(rows) < train_samples:
            print(f"Warning: train set has {len(rows)} samples, using all available.")
            selected_train = rows
        else:
            selected_train = rows[:train_samples]
        with train_30k_path.open("w", encoding="utf-8") as f:
            for r in selected_train:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Saved {len(selected_train):,} training samples to {train_30k_path}")

    if not test_30k_path.exists():
        print(f"Preparing minimum {test_samples:,} test samples from {test_full}...")
        rows = [json.loads(line) for line in test_full.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(rows) < test_samples:
            print(f"Warning: test set has {len(rows)} samples, using all available.")
            selected_test = rows
        else:
            selected_test = rows[:test_samples]
        with test_30k_path.open("w", encoding="utf-8") as f:
            for r in selected_test:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Saved {len(selected_test):,} test samples to {test_30k_path}")

    return train_30k_path, test_30k_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=10, help="Epochs per sample (default: 10)")
    parser.add_argument("--train-samples", type=int, default=30000, help="Training sample count (default: 30,000)")
    parser.add_argument("--test-samples", type=int, default=30000, help="Minimum test samples (default: 30,000)")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size per forward pass (default: 2 for 24GB VRAM)")
    parser.add_argument("--grad-accumulation", type=int, default=4, help="Gradient accumulation steps (default: 4, effective batch 8)")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--wait-for-download", action="store_true", default=False,
                        help="Poll until setup_full_vqa_dataset finishes before running")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "qwen-vqa-30k-10epochs")
    args = parser.parse_args()

    processed_dir = ROOT / "data" / "processed"
    train_manifest = processed_dir / "train.jsonl"
    val_manifest = processed_dir / "validation.jsonl"
    test_manifest = processed_dir / "locked_test.jsonl"

    if args.wait_for_download:
        print("Waiting for dataset download to complete...", flush=True)
        report_file = processed_dir / "dataset_setup_report.json"
        while not (report_file.exists() and train_manifest.exists()):
            time.sleep(30)
            print("Still waiting for dataset setup to finish...", flush=True)
        print("Dataset setup completed! Proceeding with experiment.", flush=True)

    if not train_manifest.exists() or not test_manifest.exists():
        raise FileNotFoundError(f"Manifests missing. Required: {train_manifest}, {test_manifest}")

    train_30k, test_30k = prepare_30k_splits(
        train_manifest, test_manifest, processed_dir / "splits_30k",
        train_samples=args.train_samples, test_samples=args.test_samples
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_adapter = args.output_dir / "best_adapter"
    test_predictions_out = args.output_dir / "locked_test_30k_predictions.json"
    baseline_predictions_out = args.output_dir / "baseline_locked_test_30k_predictions.json"

    # Step 1: Optimized LoRA Training (10 epochs, 30,000 samples)
    train_cmd = [
        PYTHON, str(ROOT / "scripts" / "train_qwen_vl_lora.py"),
        "--train", str(train_30k),
        "--validation", str(val_manifest),
        "--output-dir", str(args.output_dir),
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--grad-accumulation", str(args.grad_accumulation),
        "--lr", str(args.lr),
    ]
    run_cmd(train_cmd, f"LoRA Fine-Tuning Qwen2.5-VL-7B ({args.train_samples:,} samples, {args.epochs} epochs)")

    # Step 2: Evaluate Unadapted Base Model on minimum 30K test samples
    eval_base_cmd = [
        PYTHON, str(ROOT / "scripts" / "evaluate_qwen_vl.py"),
        "--data", str(test_30k),
        "--output", str(baseline_predictions_out),
    ]
    run_cmd(eval_base_cmd, f"Evaluating Unadapted Base Qwen2.5-VL on {args.test_samples:,} Test Samples")

    # Step 3: Evaluate LoRA Adapted Model on minimum 30K test samples
    eval_lora_cmd = [
        PYTHON, str(ROOT / "scripts" / "evaluate_qwen_vl.py"),
        "--data", str(test_30k),
        "--adapter", str(best_adapter),
        "--output", str(test_predictions_out),
    ]
    run_cmd(eval_lora_cmd, f"Evaluating LoRA Fine-Tuned Qwen2.5-VL on {args.test_samples:,} Test Samples")

    # Step 4: Summary Comparison Report
    base_summary = json.loads(baseline_predictions_out.read_text(encoding="utf-8")).get("summary", {})
    lora_summary = json.loads(test_predictions_out.read_text(encoding="utf-8")).get("summary", {})

    print("\n" + "="*75)
    print(f"EXPERIMENT RESULTS: 30K SAMPLES x 10 EPOCHS BENCHMARK")
    print("="*75)
    for d in ["gqa", "vqav2"]:
        if d in base_summary and d in lora_summary:
            b_acc = base_summary[d]["accuracy"]
            l_acc = lora_summary[d]["accuracy"]
            diff = l_acc - b_acc
            n = base_summary[d]["n"]
            print(f"  {d.upper()}: Baseline = {b_acc:.4f} | 10-Epoch LoRA = {l_acc:.4f} | Delta = {diff:+.4f} (N={n:,})")

    final_report = {
        "config": {
            "train_samples": args.train_samples,
            "test_samples": args.test_samples,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "grad_accumulation": args.grad_accumulation,
            "effective_batch_size": args.batch_size * args.grad_accumulation,
            "lr": args.lr,
        },
        "baseline_summary": base_summary,
        "lora_summary": lora_summary,
    }
    report_file = args.output_dir / "experiment_30k_report.json"
    report_file.write_text(json.dumps(final_report, indent=2), encoding="utf-8")
    print(f"\nSaved full benchmark report to: {report_file}")
    print("EXPERIMENT EXECUTION COMPLETE!")


if __name__ == "__main__":
    main()
