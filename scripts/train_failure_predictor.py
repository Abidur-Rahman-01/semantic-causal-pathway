"""Fit and evaluate the train-only failure predictor on disjoint image splits.

Input JSONL records are experiment result rows enriched with a ground-truth
`future_failure` label and official `split` value (train/validation/test).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from semantic_circuits.evaluation import binary_metrics, grouped_bootstrap_metric
from semantic_circuits.failure_prediction import (
    DEFAULT_FEATURES, fit_failure_predictor, flatten_pathway_features,
    score_failure_predictor, validation_threshold,
)


def read_rows(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8-sig") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
    return rows


def _group(row: dict) -> str:
    return str(row.get("image_id", row.get("sample_id", "")))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="Merged JSONL with train/validation/test rows and future_failure labels")
    parser.add_argument("--output-dir", type=Path, default=Path(".venv/outputs/failure_predictor"))
    parser.add_argument("--threshold-objective", choices=("youden",), default="youden")
    parser.add_argument("--bootstrap-repeats", type=int, default=1000)
    args = parser.parse_args()
    rows = read_rows(args.dataset)
    splits = {name: [row for row in rows if row.get("split") == name]
              for name in ("train", "validation", "test")}
    if any(not split_rows for split_rows in splits.values()):
        raise ValueError("Dataset must include nonempty train, validation, and test splits")
    group_sets = {name: {_group(row) for row in values} for name, values in splits.items()}
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlap = group_sets[left] & group_sets[right]
        if overlap:
            raise ValueError(f"Image-group leakage between {left} and {right}; example: {sorted(overlap)[0]}")
    for split_name, values in splits.items():
        if any(row.get("future_failure") not in (0, 1, False, True) for row in values):
            raise ValueError(f"Every {split_name} row needs an explicit binary future_failure label")
    if len({int(row["future_failure"]) for row in splits["validation"]}) < 2:
        raise ValueError("Validation needs both failure classes to select a threshold")

    features = {name: [flatten_pathway_features(row, row["future_failure"], row.get("sample_id"))
                       for row in values] for name, values in splits.items()}
    feature_columns = [column for column in DEFAULT_FEATURES
                       if any(row.get(column) is not None for row in features["train"])]
    if not feature_columns:
        raise ValueError("Training rows contain no pathway features")
    model, model_info = fit_failure_predictor(features["train"], feature_columns=feature_columns)
    validation_labels = [int(row["future_failure"]) for row in splits["validation"]]
    validation_scores = score_failure_predictor(model, features["validation"], feature_columns)
    threshold = validation_threshold(validation_labels, validation_scores, args.threshold_objective)

    test_labels = np.asarray([int(row["future_failure"]) for row in splits["test"]], dtype=int)
    test_scores = np.asarray(score_failure_predictor(model, features["test"], feature_columns))
    predictions = (test_scores >= threshold).astype(int)
    test_groups = [_group(row) for row in splits["test"]]
    report = {
        "n_by_split": {key: len(value) for key, value in splits.items()},
        "features": model_info["features"], "train_sample_ids": model_info["train_sample_ids"],
        "threshold": threshold, "threshold_objective": args.threshold_objective,
        "validation_metrics": binary_metrics(validation_labels, validation_scores),
        "test_metrics": {
            **binary_metrics(test_labels, test_scores),
            "accuracy": float(accuracy_score(test_labels, predictions)),
            "precision": float(precision_score(test_labels, predictions, zero_division=0)),
            "recall": float(recall_score(test_labels, predictions, zero_division=0)),
            "grouped_auroc_ci95": grouped_bootstrap_metric(
                test_labels, test_scores, test_groups, "auroc", repeats=args.bootstrap_repeats),
        },
        "interpretation": "Threshold fitted on validation; test evaluated once. Predictions estimate the supplied label, not causal truth.",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / "failure_predictor.joblib"
    report_path = args.output_dir / "failure_predictor_report.json"
    joblib.dump(model, model_path)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved model: {model_path}\nSaved report: {report_path}")


if __name__ == "__main__":
    main()
