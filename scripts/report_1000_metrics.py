"""Summarize statistical metrics for the 1,000-image study from results JSONL."""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_fresh_1000_study import evaluate_split_comprehensive, read_rows


def format_table(title: str, report: dict):
    print("\n" + "=" * 80)
    print(f" {title.upper()} ")
    print("=" * 80)
    print(f"Total Cohort Instances     : {report.get('total_instances', 0)}")
    print(f"Baseline Correct (Eligible): {report.get('baseline_correct', 0)} ({report.get('baseline_accuracy', 0)*100:.1f}%)")
    counts = report.get("event_counts", {})
    print(f"Held-out Failure Events    : {counts.get('failure_events', 0)} / {counts.get('total_eligible', 0)} ({report.get('failure_prevalence', 0)*100:.1f}%)")
    inv = report.get("behavioral_invariant_count", 0)
    col = report.get("invariant_collapsed_count", 0)
    rate = report.get("invariant_collapse_rate", 0) * 100
    print(f"Behavioral Invariant Probe : {inv} instances")
    print(f"Internal Circuit Collapse  : {col} / {inv} ({rate:.1f}% with CPS < 0.50)")

    if "predictors" not in report:
        print("\n[Notice]: Insufficient events to compute AUROC / AUPRC.")
        return

    print("-" * 80)
    print(f"{'Predictor':<32} | {'AUROC':<7} | {'95% CI':<16} | {'AUPRC':<7} | {'Brier':<7}")
    print("-" * 80)
    for name, m in report["predictors"].items():
        ci = f"[{m['auroc_ci95'][0]:.3f}, {m['auroc_ci95'][1]:.3f}]"
        print(f"{name:<32} | {m['auroc']:<7.3f} | {ci:<16} | {m['auprc']:<7.3f} | {m['brier']:<7.3f}")
    print("-" * 80)


def main():
    parser = argparse.ArgumentParser(description="Evaluate current 1,000 study results.")
    parser.add_argument(
        "--results",
        type=Path,
        default=ROOT / ".venv/outputs/study_1000/models/qwen2_5_vl_7b/semantic_pathway_results.jsonl",
        help="Path to results jsonl file.",
    )
    args = parser.parse_args()

    if not args.results.exists():
        print(f"Error: Results file not found at {args.results}")
        return

    rows = list(read_rows(args.results))
    print(f"Loaded {len(rows)} evaluated instances from {args.results}")
    split_counts = Counter(r.get("analysis_split") for r in rows)
    dataset_counts = Counter(r.get("dataset") for r in rows)
    print(f"Partitions: {dict(split_counts)}")
    print(f"Datasets  : {dict(dataset_counts)}")

    if split_counts.get("train", 0) > 0:
        train_eval = evaluate_split_comprehensive(rows, "train")
        format_table("Train Split (400 Partition)", train_eval["combined"])

    if split_counts.get("validation", 0) > 0:
        val_eval = evaluate_split_comprehensive(rows, "validation")
        format_table(f"Validation Split ({split_counts['validation']}/200 Completed)", val_eval["combined"])

    if split_counts.get("test", 0) > 0:
        test_eval = evaluate_split_comprehensive(rows, "test")
        format_table(f"Locked Test Split ({split_counts['test']}/400 Completed)", test_eval["combined"])


if __name__ == "__main__":
    main()
