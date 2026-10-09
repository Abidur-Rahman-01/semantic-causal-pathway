"""Execute full 5-fold cross-validation, final training, and locked test evaluation on CUDA."""
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
    print(f"\n{'='*70}\n[RUNNING] {desc}\nCOMMAND: {' '.join(str(c) for c in cmd)}\n{'='*70}", flush=True)
    t0 = time.time()
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.run(cmd, cwd=str(ROOT), env=env)
    elapsed = time.time() - t0
    if proc.returncode != 0:
        raise RuntimeError(f"Step failed ({desc}) with returncode {proc.returncode} in {elapsed:.1f}s")
    print(f"[COMPLETED] {desc} in {elapsed:.1f}s", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs per fold and final run")
    parser.add_argument("--folds", type=int, default=5, help="Number of folds (0 to folds-1)")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accumulation", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--eval-test", action="store_true", default=True, help="Evaluate locked test set")
    parser.add_argument("--skip-train-folds", action="store_true", help="Skip fold training if already trained")
    args = parser.parse_args()

    folds_dir = ROOT / "data" / "processed" / "folds"
    output_root = ROOT / "outputs" / "qwen-vqa-lora"
    output_root.mkdir(parents=True, exist_ok=True)

    # 1. Train and evaluate all 5 folds
    fold_reports = {}
    for fold in range(args.folds):
        train_file = folds_dir / f"fold_{fold}" / "train.jsonl"
        val_file = folds_dir / f"fold_{fold}" / "validation.jsonl"
        fold_out = output_root / f"fold_{fold}"
        best_adapter = fold_out / "best_adapter"
        val_pred_file = fold_out / "validation_predictions.json"

        if not train_file.exists() or not val_file.exists():
            raise FileNotFoundError(f"Fold files missing for fold {fold}: {train_file}, {val_file}")

        # Train fold
        if not (args.skip_train_folds and best_adapter.exists()):
            train_cmd = [
                PYTHON, str(ROOT / "scripts" / "train_qwen_vl_lora.py"),
                "--train", str(train_file),
                "--validation", str(val_file),
                "--output-dir", str(fold_out),
                "--epochs", str(args.epochs),
                "--batch-size", str(args.batch_size),
                "--grad-accumulation", str(args.grad_accumulation),
                "--lr", str(args.lr),
            ]
            run_cmd(train_cmd, f"LoRA Training Fold {fold} ({args.epochs} epochs)")
        else:
            print(f"[REUSING] Existing adapter for Fold {fold}: {best_adapter}")

        # Evaluate fold on its validation split
        if not (args.skip_train_folds and val_pred_file.exists()):
            eval_cmd = [
                PYTHON, str(ROOT / "scripts" / "evaluate_qwen_vl.py"),
                "--data", str(val_file),
                "--adapter", str(best_adapter),
                "--output", str(val_pred_file),
            ]
            run_cmd(eval_cmd, f"Evaluating Fold {fold} on Validation Split")
        else:
            print(f"[REUSING] Existing validation evaluation for Fold {fold}: {val_pred_file}")

        # Load metrics
        val_data = json.loads(val_pred_file.read_text(encoding="utf-8"))
        fold_reports[f"fold_{fold}"] = val_data["summary"]

    # 2. Aggregate 5-Fold Cross-Validation Metrics
    print("\n" + "="*70)
    print("AGGREGATING 5-FOLD CROSS-VALIDATION RESULTS")
    print("="*70)
    
    import numpy as np
    datasets = ["gqa", "vqav2"]
    agg_summary = {"per_fold": fold_reports, "cross_validation": {}}
    for d in datasets:
        accs = [fold_reports[f"fold_{f}"][d]["accuracy"] for f in range(args.folds) if d in fold_reports[f"fold_{f}"]]
        counts = [fold_reports[f"fold_{f}"][d]["n"] for f in range(args.folds) if d in fold_reports[f"fold_{f}"]]
        if accs:
            mean_acc = float(np.mean(accs))
            std_acc = float(np.std(accs))
            total_samples = int(np.sum(counts))
            agg_summary["cross_validation"][d] = {
                "mean_accuracy": mean_acc,
                "std_accuracy": std_acc,
                "fold_accuracies": accs,
                "total_evaluated_samples": total_samples,
            }
            print(f"  {d.upper()}: Mean Accuracy = {mean_acc:.4f} +/- {std_acc:.4f} (across {len(accs)} folds, N={total_samples})")

    # Overall accuracy
    all_fold_overall = []
    for f in range(args.folds):
        f_dict = fold_reports[f"fold_{f}"]
        tot_correct = sum(f_dict[d]["accuracy"] * f_dict[d]["n"] for d in f_dict)
        tot_n = sum(f_dict[d]["n"] for d in f_dict)
        all_fold_overall.append(tot_correct / max(tot_n, 1))
    
    agg_summary["cross_validation"]["overall"] = {
        "mean_accuracy": float(np.mean(all_fold_overall)),
        "std_accuracy": float(np.std(all_fold_overall)),
        "fold_accuracies": all_fold_overall,
    }
    print(f"  OVERALL: Mean Accuracy = {np.mean(all_fold_overall):.4f} +/- {np.std(all_fold_overall):.4f}")

    cv_summary_path = output_root / "cross_validation_summary.json"
    cv_summary_path.write_text(json.dumps(agg_summary, indent=2), encoding="utf-8")
    print(f"\nSaved cross-validation summary: {cv_summary_path}")

    # 3. Train final adapter on full train.jsonl
    full_train = ROOT / "data" / "processed" / "train.jsonl"
    full_val = ROOT / "data" / "processed" / "validation.jsonl"
    final_out = output_root / "final"
    final_best_adapter = final_out / "best_adapter"

    final_train_cmd = [
        PYTHON, str(ROOT / "scripts" / "train_qwen_vl_lora.py"),
        "--train", str(full_train),
        "--validation", str(full_val),
        "--output-dir", str(final_out),
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--grad-accumulation", str(args.grad_accumulation),
        "--lr", str(args.lr),
    ]
    run_cmd(final_train_cmd, f"Training Final LoRA Model on full train.jsonl ({args.epochs} epochs)")

    # 4. Evaluate on full locked_test.jsonl (181 samples)
    if args.eval_test:
        locked_test = ROOT / "data" / "processed" / "locked_test.jsonl"
        baseline_test_pred = ROOT / "outputs" / "baseline_test_predictions.json"
        final_test_pred = final_out / "locked_test_predictions.json"

        # Baseline evaluation
        eval_base_cmd = [
            PYTHON, str(ROOT / "scripts" / "evaluate_qwen_vl.py"),
            "--data", str(locked_test),
            "--output", str(baseline_test_pred),
        ]
        run_cmd(eval_base_cmd, "Evaluating Unadapted Base Qwen2.5-VL on locked_test.jsonl (Full 181 samples)")

        # Fine-tuned evaluation
        eval_final_cmd = [
            PYTHON, str(ROOT / "scripts" / "evaluate_qwen_vl.py"),
            "--data", str(locked_test),
            "--adapter", str(final_best_adapter),
            "--output", str(final_test_pred),
        ]
        run_cmd(eval_final_cmd, "Evaluating Final LoRA Model on locked_test.jsonl (Full 181 samples)")

        base_res = json.loads(baseline_test_pred.read_text(encoding="utf-8"))["summary"]
        lora_res = json.loads(final_test_pred.read_text(encoding="utf-8"))["summary"]

        print("\n" + "="*70)
        print("LOCKED TEST BENCHMARK COMPARISON (FULL 181 SAMPLES)")
        print("="*70)
        for d in datasets:
            if d in base_res and d in lora_res:
                b_acc = base_res[d]["accuracy"]
                l_acc = lora_res[d]["accuracy"]
                diff = l_acc - b_acc
                print(f"  {d.upper()}: Baseline = {b_acc:.4f} | LoRA = {l_acc:.4f} | Diff = {diff:+.4f} (N={base_res[d]['n']})")

    print("\nALL EXPERIMENTS SUCCESSFULLY EXECUTED END TO END ON CUDA!")


if __name__ == "__main__":
    main()
